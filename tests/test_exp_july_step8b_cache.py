import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import (
    _TRAJECTORY_VALIDATION_THRESHOLDS,
    step8b_causal_filter_out,
)


class Step8BCacheTests(unittest.TestCase):
    @staticmethod
    def _empty_generated_evidence():
        return {
            "version": 2,
            "evidence_type": "trajectory_motion_evidence",
            "video_id": "demo",
            "num_frames": 3,
            "num_trajectories": 0,
            "num_observations": 0,
            "trajectory_motion_evidence": [],
        }

    def test_existing_video_cache_bypasses_trajectory_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_root = Path(tmp)
            output_root = pipeline_root / "08b_test"
            cache_dir = output_root / "demo"
            cache_dir.mkdir(parents=True)
            fingerprint = hashlib.sha256(
                json.dumps(
                    _TRAJECTORY_VALIDATION_THRESHOLDS,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            protection_fingerprint = hashlib.sha256(b"{}").hexdigest()
            cached = {
                "version": 2,
                "evidence_type": "trajectory_motion_evidence",
                "video_id": "demo",
                "threshold_policy_version": 1,
                "threshold_policy_fingerprint": fingerprint,
                "semantic_protection_fingerprint": protection_fingerprint,
                "num_frames": 3,
                "num_trajectories": 1,
                "num_valid_trajectories": 1,
                "num_observations": 3,
                "trajectory_motion_evidence": [{"track_id": 7}],
            }
            (cache_dir / "trajectory_motion_evidence.json").write_text(
                json.dumps(cached), encoding="utf-8"
            )
            state = {
                "videos": ["demo"],
                "relative_object_motion": [{
                    "video_id": "demo", "num_frames": 3, "frames": []
                }],
            }
            old_root = os.environ.get("CAUVID_PIPELINE_OUTPUT_PATH")
            os.environ["CAUVID_PIPELINE_OUTPUT_PATH"] = str(pipeline_root)
            try:
                with patch(
                    "src.exp_july.perception.pipeline._trajectory_motion_evidence_video",
                    side_effect=AssertionError("expensive validation must not run"),
                ):
                    result = step8b_causal_filter_out(
                        {"ego_motion": []}, state,
                        output_subdir="08b_test", step_label="8b",
                    )
            finally:
                if old_root is None:
                    os.environ.pop("CAUVID_PIPELINE_OUTPUT_PATH", None)
                else:
                    os.environ["CAUVID_PIPELINE_OUTPUT_PATH"] = old_root

            self.assertEqual(result["trajectory_motion_evidence"], [cached])
            manifest = json.loads(
                (output_root / "trajectory_motion_evidence_manifest.json").read_text()
            )
            self.assertEqual(manifest["num_videos"], 1)
            self.assertEqual(manifest["num_trajectories"], 1)

    def test_changed_threshold_fingerprint_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_root = Path(tmp)
            cache_dir = pipeline_root / "08b_test" / "demo"
            cache_dir.mkdir(parents=True)
            cached = {
                "version": 2,
                "evidence_type": "trajectory_motion_evidence",
                "video_id": "demo",
                "threshold_policy_version": 1,
                "threshold_policy_fingerprint": "stale",
                "semantic_protection_fingerprint": hashlib.sha256(b"{}").hexdigest(),
                "trajectory_motion_evidence": [],
            }
            (cache_dir / "trajectory_motion_evidence.json").write_text(
                json.dumps(cached), encoding="utf-8"
            )
            state = {
                "videos": ["demo"],
                "relative_object_motion": [
                    {"video_id": "demo", "num_frames": 3, "frames": []}
                ],
            }
            with patch.dict(
                os.environ, {"CAUVID_PIPELINE_OUTPUT_PATH": str(pipeline_root)}
            ), patch(
                "src.exp_july.perception.pipeline._trajectory_motion_evidence_video",
                return_value=self._empty_generated_evidence(),
            ) as generate:
                step8b_causal_filter_out(
                    {"ego_motion": []},
                    state,
                    output_subdir="08b_test",
                    step_label="8b",
                )
            generate.assert_called_once()

    def test_changed_semantic_protection_invalidates_final_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_root = Path(tmp)
            cache_dir = pipeline_root / "08f_test" / "demo"
            cache_dir.mkdir(parents=True)
            threshold_fingerprint = hashlib.sha256(
                json.dumps(
                    _TRAJECTORY_VALIDATION_THRESHOLDS,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            cached = {
                "version": 2,
                "evidence_type": "trajectory_motion_evidence",
                "video_id": "demo",
                "threshold_policy_version": 1,
                "threshold_policy_fingerprint": threshold_fingerprint,
                "semantic_protection_fingerprint": hashlib.sha256(b"{}").hexdigest(),
                "trajectory_motion_evidence": [],
            }
            (cache_dir / "trajectory_motion_evidence.json").write_text(
                json.dumps(cached), encoding="utf-8"
            )
            state = {
                "videos": ["demo"],
                "relative_object_motion": [
                    {"video_id": "demo", "num_frames": 3, "frames": []}
                ],
                "protected_objects": [{
                    "video_id": "demo",
                    "track_id": 7,
                    "matched_rule_ids": ["protect"],
                }],
            }
            with patch.dict(
                os.environ, {"CAUVID_PIPELINE_OUTPUT_PATH": str(pipeline_root)}
            ), patch(
                "src.exp_july.perception.pipeline._trajectory_motion_evidence_video",
                return_value=self._empty_generated_evidence(),
            ) as generate:
                step8b_causal_filter_out(
                    {"ego_motion": []},
                    state,
                    phase="final",
                    output_subdir="08f_test",
                    step_label="8f",
                )
            generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
