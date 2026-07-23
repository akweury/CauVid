import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.uncertain_signal_evidence_visualization import (
    render_step8b_signal_evidence_videos,
    select_step8b_visualization_tracks,
)


def _evidence_video(video_id, confidences):
    return {
        "video_id": video_id,
        "track_signal_evidence": [
            {
                "track_id": track_id,
                "primary_label": "car",
                "evidence_confidence": confidence,
                "descriptors": {},
            }
            for track_id, confidence in confidences.items()
        ],
    }


class Step8BVisualizationTests(unittest.TestCase):
    def test_selection_is_deterministic_capped_and_prefers_uncertain_tracks(self):
        evidence = [
            _evidence_video(
                "scene",
                {1: 0.92, 2: 0.18, 3: 0.51, 4: 0.09, 5: 0.31},
            )
        ]
        selected = select_step8b_visualization_tracks(
            evidence, max_tracks_per_video=99
        )
        reversed_selected = select_step8b_visualization_tracks(
            [
                {
                    **evidence[0],
                    "track_signal_evidence": list(
                        reversed(evidence[0]["track_signal_evidence"])
                    ),
                }
            ],
            max_tracks_per_video=3,
        )

        self.assertEqual([row["track_id"] for row in selected], [4, 2, 5])
        self.assertEqual(
            [row["track_id"] for row in reversed_selected],
            [4, 2, 5],
        )
        self.assertEqual(len(selected), 3)

    def test_renderer_writes_three_track_manifest_and_evidence_files(self):
        evidence = [
            _evidence_video(
                "scene",
                {1: 0.92, 2: 0.18, 3: 0.51, 4: 0.09, 5: 0.31},
            )
        ]
        relative_motion = [
            {
                "video_id": "scene",
                "frames": [{"frame_index": 0, "objects": []}],
            }
        ]

        def fake_render(_video, _evidence, output_path, **_kwargs):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return str(output_path), "rendered"

        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception."
            "uncertain_signal_evidence_visualization."
            "_render_step8b_track_video",
            side_effect=fake_render,
        ) as render:
            stale_root = Path(tmp) / "scene" / "track_0001"
            stale_root.mkdir(parents=True)
            stale_video = stale_root / "track_0001_step8b_evidence.mp4"
            stale_video.touch()
            (stale_root / "track_0001_evidence.json").write_text("{}")
            result = render_step8b_signal_evidence_videos(
                relative_motion,
                evidence,
                Path(tmp),
                max_tracks_per_video=8,
            )

            self.assertEqual(render.call_count, 3)
            self.assertEqual(result["max_tracks_per_video"], 3)
            self.assertEqual(result["num_selected_tracks"], 3)
            self.assertEqual(result["num_rendered_videos"], 3)
            self.assertEqual(result["num_pruned_stale_artifacts"], 2)
            self.assertFalse(stale_video.exists())
            self.assertEqual(
                result["selections"][0]["selected_track_ids"],
                [4, 2, 5],
            )
            self.assertEqual(
                result["confidence_color_scale"],
                {"low": "red", "medium": "yellow", "high": "green"},
            )
            self.assertTrue(Path(result["manifest_path"]).exists())
            self.assertEqual(
                len(list(Path(tmp).glob("scene/track_*/*.mp4"))),
                3,
            )
            self.assertEqual(
                len(list(Path(tmp).glob("scene/track_*/*_evidence.json"))),
                3,
            )


if __name__ == "__main__":
    unittest.main()
