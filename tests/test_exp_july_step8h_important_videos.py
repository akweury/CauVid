import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import step8h_visual_relative_motion


class Step8HImportantVideoTests(unittest.TestCase):
    def test_only_videos_with_semantically_important_objects_are_rendered(self):
        state = {
            "videos": ["ordinary", "important"],
            "relative_object_motion": [
                {"video_id": "ordinary", "frames": []},
                {"video_id": "important", "frames": []},
            ],
            "important_objects": [
                {"video_id": "important", "track_id": 7}
            ],
            "protected_objects": [],
            "trajectory_motion_evidence": [],
            "refined_ego_motion": [],
        }

        def render_video(*, relative_motion_video_result, **_kwargs):
            return ([{"video_id": relative_motion_video_result["video_id"]}], [])

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"CAUVID_PIPELINE_OUTPUT_PATH": tmp}
        ), patch(
            "src.exp_july.perception.pipeline._save_ego_motion_comparison_pdf",
            return_value=(None, "not_available"),
        ), patch(
            "src.exp_july.perception.pipeline._render_relative_motion_track_videos",
            side_effect=render_video,
        ) as render:
            result = step8h_visual_relative_motion(state)

            self.assertEqual(render.call_count, 1)
            self.assertEqual(
                result["relative_motion_visualizations"],
                [{"video_id": "important"}],
            )
            selection = result[
                "relative_motion_visualization_video_selection"
            ]
            self.assertEqual(selection["rendered_video_ids"], ["important"])
            self.assertEqual(
                selection["skipped_unimportant_video_ids"], ["ordinary"]
            )
            manifest = json.loads(
                (
                    Path(tmp)
                    / "08h_relative_motion_tracks"
                    / "relative_motion_track_visualization_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["render_scope"], "important_videos")
            self.assertEqual(manifest["num_source_videos"], 2)
            self.assertEqual(manifest["num_videos"], 1)


if __name__ == "__main__":
    unittest.main()
