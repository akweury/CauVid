import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import (
    _uncertain_signal_evidence_video,
    step8b_uncertain_signal_evidence,
)


def _relative_video():
    frames = []
    for frame_index in range(5):
        rel_vx = 0.2 + 0.05 * frame_index
        rel_vz = -0.4 - 0.05 * frame_index
        frames.append(
            {
                "frame_index": frame_index,
                "objects": [
                    {
                        "track_id": 7,
                        "object_index": 0,
                        "frame_label": "car",
                        "label": "car",
                        "bbox": [
                            100 + frame_index,
                            80,
                            150 + frame_index,
                            130,
                        ],
                        "position_3d": [
                            0.2 * frame_index,
                            0.0,
                            12.0 - 0.4 * frame_index,
                        ],
                        "obj_vx": rel_vx,
                        "obj_vz": rel_vz,
                        "ego_vx": 0.0,
                        "ego_vz": 0.0,
                        "rel_vx": rel_vx,
                        "rel_vz": rel_vz,
                        "rel_speed": (rel_vx**2 + rel_vz**2) ** 0.5,
                        "has_rel_motion": frame_index > 0,
                        "motion_state": "observed_with_rel_motion",
                        "score": 0.95,
                        "source": "observed",
                        "source_type": "detector_track",
                        "is_observed": True,
                        "is_repaired": False,
                    }
                ],
            }
        )
    return {
        "video_id": "demo",
        "num_frames": len(frames),
        "frames": frames,
    }


def _short_track_video(
    label="car", *, depth=70.0, score=0.30, depth_step=0.0
):
    frames = []
    for frame_index in range(2):
        frames.append(
            {
                "frame_index": frame_index,
                "objects": [
                    {
                        "track_id": 11,
                        "object_index": 0,
                        "frame_label": label,
                        "label": label,
                        "bbox": [100, 80, 110, 90],
                        "position_3d": [
                            8.0,
                            0.0,
                            depth + depth_step * frame_index,
                        ],
                        "rel_vx": 0.0,
                        "rel_vz": 0.0,
                        "rel_speed": 0.0,
                        "has_rel_motion": frame_index > 0,
                        "score": score,
                        "source": "observed",
                        "source_type": "detector_track",
                    }
                ],
            }
        )
    return {"video_id": "short", "num_frames": 200, "frames": frames}


def _all_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


class Step8BSignalEvidenceTests(unittest.TestCase):
    def test_initial_filter_quarantines_only_unanimously_weak_far_vehicle(self):
        evidence = _uncertain_signal_evidence_video(_short_track_video())

        self.assertEqual(evidence["num_source_tracks"], 1)
        self.assertEqual(evidence["num_active_tracks"], 0)
        self.assertEqual(evidence["num_quarantined_tracks"], 1)
        self.assertEqual(evidence["track_signal_evidence"], [])
        quarantined = evidence["quarantined_track_signal_evidence"][0]
        self.assertEqual(
            set(quarantined),
            {"track_id", "primary_label", "observable_cues"},
        )
        decision = evidence["track_usefulness_filter"]["decisions"][0]
        self.assertEqual(decision["decision"], "quarantine")
        self.assertEqual(
            decision["reason_codes"],
            ["unanimous_short_tiny_far_weak_vehicle"],
        )
        self.assertTrue(all(decision["conditions"].values()))

    def test_initial_filter_never_quarantines_short_traffic_control(self):
        evidence = _uncertain_signal_evidence_video(
            _short_track_video("traffic sign")
        )

        self.assertEqual(evidence["num_active_tracks"], 1)
        self.assertEqual(evidence["num_quarantined_tracks"], 0)
        decision = evidence["track_usefulness_filter"]["decisions"][0]
        self.assertEqual(decision["decision"], "active")
        self.assertIn(
            "protected_semantic_category", decision["reason_codes"]
        )

    def test_initial_filter_preserves_near_or_strong_short_vehicle(self):
        near = _uncertain_signal_evidence_video(
            _short_track_video(depth=20.0)
        )
        strong = _uncertain_signal_evidence_video(
            _short_track_video(score=0.90)
        )

        self.assertEqual(near["num_active_tracks"], 1)
        self.assertIn(
            "near_ego",
            near["track_usefulness_filter"]["decisions"][0]["reason_codes"],
        )
        self.assertEqual(strong["num_active_tracks"], 1)
        self.assertIn(
            "strong_detection",
            strong["track_usefulness_filter"]["decisions"][0][
                "reason_codes"
            ],
        )

    def test_initial_filter_preserves_raw_short_approach(self):
        approaching = _uncertain_signal_evidence_video(
            _short_track_video(depth_step=-2.0)
        )

        self.assertEqual(approaching["num_active_tracks"], 1)
        self.assertIn(
            "raw_depth_approach",
            approaching["track_usefulness_filter"]["decisions"][0][
                "reason_codes"
            ],
        )

    def test_video_abstraction_emits_only_six_scalar_observable_cues(self):
        evidence = _uncertain_signal_evidence_video(_relative_video())

        self.assertEqual(evidence["evidence_type"], "uncertain_signal_evidence")
        self.assertEqual(
            evidence["abstraction_level"], "low_level_observable_signal"
        )
        self.assertFalse(evidence["semantic_motion_classification"])
        self.assertFalse(evidence["symbolic_reasoning"])
        self.assertEqual(evidence["num_tracks"], 1)
        track = evidence["track_signal_evidence"][0]
        self.assertEqual(track["track_id"], 7)
        self.assertNotIn("trajectory_observations", track)
        self.assertEqual(
            set(track),
            {"track_id", "primary_label", "observable_cues"},
        )
        self.assertEqual(
            set(track["observable_cues"]),
            {
                "leftness",
                "rightness",
                "approach",
                "recede",
                "acceleration",
                "deceleration",
            },
        )
        self.assertEqual(track["observable_cues"]["leftness"], 0.0)
        self.assertGreater(track["observable_cues"]["rightness"], 0.0)
        self.assertGreater(track["observable_cues"]["approach"], 0.0)
        self.assertEqual(track["observable_cues"]["recede"], 0.0)
        self.assertGreater(track["observable_cues"]["acceleration"], 0.0)
        self.assertEqual(track["observable_cues"]["deceleration"], 0.0)
        for cue in track["observable_cues"].values():
            self.assertGreaterEqual(cue, 0.0)
            self.assertLessEqual(cue, 1.0)

        forbidden = {
            "validation_status",
            "valid",
            "invalid",
            "fact_decision",
            "fact_decision_status",
            "motion_significance",
            "symbolic_layer_eligible",
            "trajectory_pattern",
            "driving_fact",
            "descriptors",
            "observation_quality",
            "longitudinal_trend",
            "lateral_trend",
            "speed_trend",
            "temporal_coherence",
            "signal_reference",
            "provenance",
        }
        self.assertFalse(forbidden & _all_keys(evidence))

    def test_step_writes_new_contract_and_drops_stale_decision_state(self):
        state = {
            "videos": ["demo"],
            "relative_object_motion": [_relative_video()],
            "trajectory_motion_evidence": [{"legacy": True}],
            "trajectory_motion_evidence_phase": "repaired",
            "causal_filter_out": [{"legacy": True}],
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
        ):
            result = step8b_uncertain_signal_evidence(state)
            root = Path(tmp) / "08b_uncertain_signal_evidence"
            payload = json.loads(
                (
                    root / "demo" / "uncertain_signal_evidence.json"
                ).read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (
                    root / "uncertain_signal_evidence_manifest.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(
            result["step8b_evidence_type"], "uncertain_signal_evidence"
        )
        self.assertNotIn("trajectory_motion_evidence", result)
        self.assertNotIn("causal_filter_out", result)
        self.assertEqual(payload["num_tracks"], 1)
        self.assertEqual(manifest["num_tracks"], 1)
        self.assertEqual(manifest["num_source_tracks"], 1)
        self.assertEqual(manifest["num_quarantined_tracks"], 0)
        self.assertEqual(
            manifest["track_usefulness_filter"]["policy_version"], 1
        )
        self.assertFalse(manifest["semantic_motion_classification"])
        self.assertFalse(manifest["symbolic_reasoning"])
        self.assertEqual(
            manifest["cue_names"],
            [
                "leftness",
                "rightness",
                "approach",
                "recede",
                "acceleration",
                "deceleration",
            ],
        )
        self.assertEqual(
            manifest["visualization"]["max_tracks_per_video"], 3
        )
        self.assertEqual(
            manifest["visualization"]["selection_policy"],
            "step8b-lowest-confidence-first-v1",
        )
        self.assertEqual(
            result[
                "uncertain_signal_evidence_visualization_manifest"
            ]["num_selected_tracks"],
            1,
        )

    def test_cache_is_invalidated_when_source_signal_changes(self):
        video = _relative_video()
        state = {
            "videos": ["demo"],
            "relative_object_motion": [video],
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
        ):
            first = step8b_uncertain_signal_evidence(state)
            first_fingerprint = first["uncertain_signal_evidence"][0][
                "source_signal_fingerprint"
            ]
            video["frames"][-1]["objects"][0]["position_3d"][2] -= 2.0
            second = step8b_uncertain_signal_evidence(state)
            second_fingerprint = second["uncertain_signal_evidence"][0][
                "source_signal_fingerprint"
            ]

        self.assertNotEqual(first_fingerprint, second_fingerprint)


if __name__ == "__main__":
    unittest.main()
