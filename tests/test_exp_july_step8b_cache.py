import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import step8b_causal_filter_out


class Step8BCacheTests(unittest.TestCase):
    def test_existing_video_cache_bypasses_trajectory_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_root = Path(tmp)
            output_root = pipeline_root / "08b_test"
            cache_dir = output_root / "demo"
            cache_dir.mkdir(parents=True)
            cached = {
                "version": 1,
                "evidence_type": "trajectory_motion_evidence",
                "video_id": "demo",
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


if __name__ == "__main__":
    unittest.main()
