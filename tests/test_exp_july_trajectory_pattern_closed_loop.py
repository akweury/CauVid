import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.adaptive_motion_repair import _evaluate, _recompute_motion
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
    ev = _evaluate(obs, 6)
    trajectory = {
        "track_id": 7,
        "primary_label": "car",
        "trajectory_observations": obs,
        "trajectory_statistics": ev["statistics"],
        "uncertainty": ev["uncertainty"],
        "provenance": ev["decision"]["provenance_summary"],
        "causal_motion_fact_validation": ev["validation"],
        "fact_decision_status": ev["decision"]["decision"],
    }
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
    return {
        "videos": ["demo"],
        "trajectory_motion_evidence_phase": "repaired",
        "trajectory_motion_evidence": [
            {
                "video_id": "demo",
                "num_frames": 6,
                "trajectory_motion_evidence": [trajectory],
            }
        ],
        "relative_object_motion": [
            {"video_id": "demo", "num_frames": 6, "frames": frames}
        ],
        "ego_motion": [{"video_id": "demo", "frames": []}],
    }


def llm(kind, _prompt):
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
            self.assertEqual(result["trajectory_pattern_manifest"]["interval_review_count"], 1)
            self.assertTrue(result["trajectory_pattern_manifest"]["policy_frozen"])
            self.assertTrue((root / "policies" / "active_policy.json").exists())
            self.assertTrue((root / "policies" / "epoch_0001.json").exists())
            self.assertTrue(list((root / "epoch_reviews").glob("*_package.json")))
            track_report = (
                root
                / "visualizations"
                / "demo"
                / "track_0007_pattern_process.html"
            )
            summary_report = (
                root
                / "visualizations"
                / "demo"
                / "video_pattern_summary.html"
            )
            self.assertTrue(track_report.exists())
            self.assertTrue(summary_report.exists())
            self.assertFalse(
                (
                    root
                    / "visualizations"
                    / "demo"
                    / "track_0007_pattern_process.png"
                ).exists()
            )
            self.assertFalse(
                (
                    root
                    / "visualizations"
                    / "demo"
                    / "video_pattern_summary.png"
                ).exists()
            )
            track_html = track_report.read_text(encoding="utf-8")
            for marker in (
                'id="symbolic-track"',
                'id="pattern-residuals"',
                'id="llm-interpretation"',
                'id="repair-candidates"',
                'id="symbolic-validation"',
                'id="final-result"',
                'id="provenance"',
            ):
                self.assertIn(marker, track_html)
            self.assertNotIn("https://", track_html)
            summary_html = summary_report.read_text(encoding="utf-8")
            self.assertIn("Step 8C video pattern summary", summary_html)
            self.assertIn("track_0007_pattern_process.html", summary_html)
            self.assertNotIn("https://", summary_html)
            visualization_manifest_path = (
                root
                / "visualizations"
                / "trajectory_pattern_visualization_manifest.json"
            )
            self.assertTrue(visualization_manifest_path.exists())
            visualization_manifest = json.loads(
                visualization_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(visualization_manifest["report_format"], "html")
            self.assertEqual(visualization_manifest["num_track_reports"], 1)
            self.assertEqual(visualization_manifest["num_summary_reports"], 1)
            self.assertTrue(
                visualization_manifest["track_reports"][0][
                    "visualization_path"
                ].endswith(".html")
            )
            dashboard = root / "dashboard" / "index.html"
            self.assertTrue(dashboard.exists())
            html = dashboard.read_text(encoding="utf-8")
            for marker in (
                "Step 8C Static Audit Dashboard", "Raw trajectory playback",
                "Repaired trajectory playback", "Interactive signals",
                "Pattern and residual comparison", "Repair candidates",
                "Symbolic validation", "LLM audit records",
                "Dataset-level ablation summary", "READ ONLY",
            ):
                self.assertIn(marker, html)
            self.assertNotIn("https://", html)
            self.assertTrue((root / "dashboard" / "dashboard_manifest.json").exists())

        record = result["trajectory_pattern_records"][0]
        self.assertEqual(len(record["pattern_candidates"]), len(PATTERNS))
        for candidate in record["pattern_candidates"]:
            self.assertEqual(set(candidate["residual_vector"]), set(RESIDUALS))
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
        self.assertIn("mandatory_unknown_baseline", sources)
        self.assertIn("minimum_residual_baseline", sources)
        self.assertIn("LLM_preferred_pattern", record)
        self.assertEqual(record["provenance"]["frozen_policy_version"], 1)
        self.assertEqual(record["provenance"]["epoch_id"], 1)
        self.assertIn("validated_pattern", record)
        self.assertIn(record["final_selection_reason"], {
            "highest_ranked_after_hard_constraints",
            "no_candidate_passed_hard_constraints_original_preserved",
        })
        self.assertEqual(
            result["pre_pattern_relative_object_motion"],
            source["relative_object_motion"],
        )
        row = result["trajectory_pattern_statistics_candidate"]["rows"][0]
        required = {
            "dataset", "video", "object_class", "trajectory_pattern",
            "residual_type", "sample_count", "mean", "std", "median",
            "quantiles", "accepted_count", "rejected_count",
            "repair_success_rate", "false_match_rate", "LLM_assessment", "version",
        }
        self.assertTrue(required.issubset(row))
        self.assertEqual(len(result["trajectory_pattern_visualizations"]), 1)
        self.assertEqual(len(result["trajectory_pattern_video_summaries"]), 1)


    def test_statistics_update_and_promotion_splits_are_video_disjoint(self):
        source = state()
        source["videos"].append("demo_validation")
        for key in ("trajectory_motion_evidence", "relative_object_motion", "ego_motion"):
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
