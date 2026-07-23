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
    def test_video_abstraction_emits_only_low_level_signal_descriptors(self):
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
            set(track["descriptors"]),
            {
                "observation_quality",
                "longitudinal_trend",
                "lateral_trend",
                "speed_trend",
                "temporal_coherence",
            },
        )
        self.assertEqual(
            track["descriptors"]["longitudinal_trend"]["state"],
            "decreasing",
        )
        self.assertEqual(
            track["descriptors"]["longitudinal_trend"][
                "velocity_signal"
            ]["level"],
            "negative",
        )
        self.assertEqual(
            track["descriptors"]["lateral_trend"]["state"],
            "increasing",
        )
        self.assertEqual(
            track["descriptors"]["temporal_coherence"]["state"],
            "continuous_samples",
        )
        for descriptor in track["descriptors"].values():
            self.assertGreaterEqual(descriptor["confidence"], 0.0)
            self.assertLessEqual(descriptor["confidence"], 1.0)

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
        self.assertFalse(manifest["semantic_motion_classification"])
        self.assertFalse(manifest["symbolic_reasoning"])
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
