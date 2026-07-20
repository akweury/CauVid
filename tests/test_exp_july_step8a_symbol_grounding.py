import json
import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.pipeline import _trajectory_motion_evidence_video
from src.exp_july.perception.symbol_grounded_refinement import (
    build_grounded_tracks,
    run_symbol_grounded_refinement,
)


def _obj(track_id, label, bbox, frame_index, rel_speed=0.1):
    return {
        "track_id": track_id,
        "frame_label": label,
        "label": label,
        "bbox": bbox,
        "box": bbox,
        "position_3d": [0.0, 0.0, 10.0],
        "relative_position_3d": [0.0, 0.0, 10.0],
        "rel_vx": rel_speed,
        "rel_vz": rel_speed,
        "rel_speed": rel_speed,
        "has_rel_motion": True,
        "distance_state": "near",
        "x_position_state": "centered",
        "vx_state": "vx_stable",
        "vz_state": "vz_stable",
        "speed_state": "rel_moving",
        "motion_state": "observed_with_rel_motion",
        "score": 0.9,
        "source": "observed",
        "source_type": "accepted_track",
        "is_observed": True,
        "is_repaired": False,
        "frame_index": frame_index,
    }


def _video():
    frames = []
    for frame_index in range(4):
        frames.append(
            {
                "frame_index": frame_index,
                "image_path": "",
                "objects": [
                    _obj(1, "pedestrian", [40, 20, 70, 90], frame_index),
                    _obj(2, "car", [150, 30, 240, 110], frame_index),
                ],
            }
        )
    return {"video_id": "demo", "num_frames": 4, "frames": frames}


class Step8ASymbolGroundingTests(unittest.TestCase):
    def test_valid_grounded_rule_executes(self):
        video = _video()
        tracks = build_grounded_tracks(video)
        pedestrian = next(row for row in tracks if row["track_id"] == 1)
        support = next(atom for atom in pedestrian["atoms"] if atom.startswith("object_class("))

        def llm(_prompt):
            return {
                "rules": [
                    {
                        "rule_id": "protect_pedestrian",
                        "target": "protected_object",
                        "conditions": [{"predicate": "object_class", "value": "pedestrian"}],
                        "supporting_atoms": [support],
                        "justification": "object_class is pedestrian in the supplied symbol.",
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_symbol_grounded_refinement(
                {"videos": [], "relative_object_motion": [video]},
                Path(tmp),
                llm_generate=llm,
            )
            payload = json.loads(
                (Path(tmp) / "demo" / "symbol_grounded_refinement.json").read_text()
            )
        self.assertEqual(
            set(payload),
            {
                "status",
                "rule_head_predicate",
                "grounded_tracks",
                "semantic_protection_rules",
                "rejected_rules",
                "important_objects",
                "protected_objects",
                "uncovered_track_ids",
            },
        )
        self.assertEqual([row["track_id"] for row in result["protected_objects"]], [1])
        self.assertEqual([row["track_id"] for row in result["important_objects"]], [1])
        self.assertEqual(result["symbol_grounded_refinement"][0]["uncovered_track_ids"], [2])
        self.assertEqual(len(result["semantic_protection_rules"]), 1)
        protected = result["protected_objects"][0]
        self.assertEqual(protected["matched_rule_ids"], ["protect_pedestrian"])
        self.assertTrue(protected["grounding_evidence"])
        self.assertEqual(protected["trajectory_decision"], "pending_step8b")

    def test_invented_predicate_and_threshold_are_rejected(self):
        video = _video()
        support = build_grounded_tracks(video)[0]["atoms"][0]

        def llm(_prompt):
            return {
                "rules": [
                    {
                        "rule_id": "invented",
                        "target": "protected_object",
                        "conditions": [{"predicate": "distance_less_than", "value": "5m"}],
                        "supporting_atoms": [support],
                        "justification": "Invented numeric threshold.",
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_symbol_grounded_refinement(
                {"videos": [], "relative_object_motion": [video]},
                Path(tmp),
                llm_generate=llm,
            )
        video_result = result["symbol_grounded_refinement"][0]
        self.assertEqual(video_result["semantic_protection_rules"], [])
        self.assertIn("unknown_predicate", video_result["rejected_rules"][0]["rejection_reasons"])
        self.assertEqual(result["protected_objects"], [])

    def test_redundant_and_overly_general_rules_are_rejected(self):
        video = _video()
        tracks = build_grounded_tracks(video)
        pedestrian_support = next(
            atom
            for atom in tracks[0]["atoms"]
            if atom.startswith("object_class(")
        )
        confidence_support = next(
            atom
            for atom in tracks[0]["atoms"]
            if atom.startswith("detection_confidence(")
        )

        def llm(_prompt):
            return {
                "rules": [
                    {
                        "rule_id": "pedestrian_one",
                        "target": "protected_object",
                        "conditions": [{"predicate": "object_class", "value": "pedestrian"}],
                        "supporting_atoms": [pedestrian_support],
                        "justification": "Grounded pedestrian class.",
                    },
                    {
                        "rule_id": "pedestrian_duplicate",
                        "target": "protected_object",
                        "conditions": [{"predicate": "object_class", "value": "pedestrian"}],
                        "supporting_atoms": [pedestrian_support],
                        "justification": "Duplicate grounded pedestrian class.",
                    },
                    {
                        "rule_id": "all_high_confidence",
                        "target": "protected_object",
                        "conditions": [{"predicate": "detection_confidence", "value": "high"}],
                        "supporting_atoms": [confidence_support],
                        "justification": "Grounded high confidence.",
                    },
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_symbol_grounded_refinement(
                {"videos": [], "relative_object_motion": [video]},
                Path(tmp),
                llm_generate=llm,
            )
        rejected = result["symbol_grounded_refinement"][0]["rejected_rules"]
        reasons = {reason for rule in rejected for reason in rule["rejection_reasons"]}
        self.assertIn("redundant_body", reasons)
        self.assertIn("overly_general_coverage", reasons)

    def test_protection_overrides_discard_but_preserves_audit_decision(self):
        frames = []
        for frame_index in range(3):
            obj = _obj(
                9,
                "pedestrian",
                [10 + frame_index * 1000, 10, 40 + frame_index * 1000, 50],
                frame_index,
                rel_speed=30.0,
            )
            frames.append({"frame_index": frame_index, "image_path": "", "objects": [obj]})
        video = {"video_id": "unsafe", "num_frames": 3, "frames": frames}
        unprotected = _trajectory_motion_evidence_video(video)
        self.assertEqual(unprotected["trajectory_motion_evidence"][0]["fact_decision_status"], "Discard")

        protected = _trajectory_motion_evidence_video(
            video,
            protected_by_track={
                9: {
                    "video_id": "unsafe",
                    "track_id": 9,
                    "matched_rule_ids": ["protect_pedestrian"],
                    "grounding_evidence": [{"rule_id": "protect_pedestrian"}],
                    "protection_reason": "grounded pedestrian rule",
                    "original_decision_before_protection": None,
                    "trajectory_decision": "pending_step8b",
                    "final_decision_after_protection": "pending_step8b",
                }
            },
        )
        row = protected["trajectory_motion_evidence"][0]
        self.assertEqual(row["fact_decision_status"], "Keep with uncertainty")
        self.assertTrue(row["symbolic_layer_eligible"])
        self.assertEqual(row["fact_decision"]["original_decision_before_protection"], "Discard")
        self.assertEqual(row["fact_decision"]["trajectory_decision"], "Discard")
        self.assertEqual(
            row["fact_decision"]["final_decision_after_protection"],
            "Keep_with_uncertainty",
        )
        self.assertTrue(row["fact_decision"]["send_to_motion_signal_refinement"])


if __name__ == "__main__":
    unittest.main()

