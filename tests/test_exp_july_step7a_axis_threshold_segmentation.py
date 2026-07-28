import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import step7_train_eval_split

from src.exp_july.perception.ego_axis_threshold_segmentation import (
    _confidence_at,
    _confidence_surface,
    render_all_video_plateau_scatter,
    render_segment_count_chart,
    segment_axis,
    segment_video,
)


class Step7AAxisThresholdSegmentationTests(unittest.TestCase):
    def test_pre_step_uses_deterministic_four_to_one_video_split(self):
        state = {"videos": [f"video-{index:02d}" for index in range(10)]}
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.exp_july.perception.pipeline.get_pipeline_output_root",
            return_value=Path(directory),
        ):
            first = step7_train_eval_split(state)
            second = step7_train_eval_split(state)
            self.assertEqual(first["step7_train_video_ids"], second["step7_train_video_ids"])
            self.assertEqual(first["step7_eval_video_ids"], second["step7_eval_video_ids"])
            self.assertEqual(len(first["step7_train_video_ids"]), 8)
            self.assertEqual(len(first["step7_eval_video_ids"]), 2)
            self.assertFalse(set(first["step7_train_video_ids"]) & set(first["step7_eval_video_ids"]))
            self.assertTrue(Path(first["step7_train_eval_split_path"]).exists())

    def test_uses_exactly_100_interior_threshold_candidates(self):
        frames = [
            {"frame_index": index, "ego_vz_smoothed": value}
            for index, value in enumerate([-4.0, -4.0, 0.0, 0.0, 4.0, 4.0])
        ]
        result = segment_axis(
            frames, "vz", ("backward", "static", "forward")
        )
        thresholds = [row["threshold"] for row in result["threshold_candidates"]]
        self.assertEqual(len(thresholds), 100)
        self.assertGreater(min(thresholds), 0.0)
        self.assertLess(max(thresholds), 4.0)

    def test_keeps_every_long_plateau_and_uses_its_middle_n(self):
        frames = [
            {"frame_index": index, "ego_vz_smoothed": value}
            for index, value in enumerate([-3.0] * 4 + [0.0] * 4 + [3.0] * 4)
        ]
        result = segment_axis(frames, "vz", ("backward", "static", "forward"))
        self.assertTrue(result["qualifying_plateaus"])
        self.assertNotIn("optimal_threshold", result)
        self.assertNotIn("selected_plateau", result)
        for plateau in result["qualifying_plateaus"]:
            self.assertGreater(plateau["num_n_values"], 5)
            self.assertGreater(plateau["segment_count"], 1)
            self.assertAlmostEqual(
                plateau["midpoint_n"],
                plateau["candidate_optimal_n"],
            )
            self.assertAlmostEqual(
                plateau["midpoint_n"],
                0.5 * (plateau["threshold_start"] + plateau["threshold_end"]),
            )

    def test_excludes_plateaus_that_map_to_one_temporal_segment(self):
        frames = [
            {"frame_index": index, "ego_vz_smoothed": 2.0}
            for index in range(20)
        ]
        result = segment_axis(frames, "vz", ("backward", "static", "forward"))
        self.assertTrue(any(row["segment_count"] == 1 for row in result["all_plateaus"]))
        self.assertTrue(all(row["segment_count"] > 1 for row in result["qualifying_plateaus"]))

    def test_vz_and_vx_use_requested_state_vocabulary(self):
        frames = []
        for index, (vx, vz) in enumerate(
            [(-2.0, -3.0)] * 4
            + [(0.0, 0.0)] * 4
            + [(2.0, 3.0)] * 4
        ):
            frames.append(
                {
                    "frame_index": index,
                    "ego_vx_smoothed": vx,
                    "ego_vz_smoothed": vz,
                }
            )
        result = segment_video({"video_id": "demo", "frames": frames})
        self.assertEqual(
            {row["state"] for row in result["vz_segmentation"]["qualifying_plateaus"][0]["segments"]},
            {"backward", "static", "forward"},
        )
        self.assertEqual(
            {row["state"] for row in result["vx_segmentation"]["qualifying_plateaus"][0]["segments"]},
            {"right", "straight", "left"},
        )


    def test_renders_individual_one_by_two_segment_count_chart(self):
        frames = [
            {
                "frame_index": index,
                "ego_vx_smoothed": float(index % 3 - 1),
                "ego_vz_smoothed": float(index % 5 - 2),
            }
            for index in range(20)
        ]
        result = segment_video({"video_id": "chart-demo", "frames": frames})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "axis_threshold_segment_counts.png"
            self.assertEqual(render_segment_count_chart(result, path), str(path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


    def test_confidence_surface_has_smooth_decay_from_training_cluster(self):
        rows = [
            {"midpoint_n": 10.0, "segment_count": 3},
            {"midpoint_n": 10.1, "segment_count": 3},
            {"midpoint_n": 10.2, "segment_count": 3},
            {"midpoint_n": 11.0, "segment_count": 3},
        ]
        model = _confidence_surface(rows)
        near = _confidence_at(model, {"midpoint_n": 10.1, "segment_count": 3})
        medium = _confidence_at(model, {"midpoint_n": 12.0, "segment_count": 3})
        far = _confidence_at(model, {"midpoint_n": 30.0, "segment_count": 6})
        self.assertGreater(near, medium)
        self.assertGreater(medium, far)
        self.assertGreaterEqual(near, 0.9)
        self.assertGreaterEqual(far, 0.0)
        self.assertEqual(model["audit"]["gradient"], "continuous_gaussian_decay")

    def test_renders_one_scatter_chart_over_all_videos(self):
        videos = []
        for video_index in range(2):
            frames = [
                {
                    "frame_index": index,
                    "ego_vx_smoothed": value * (video_index + 1),
                    "ego_vz_smoothed": value * (video_index + 2),
                }
                for index, value in enumerate([-2.0] * 5 + [0.0] * 5 + [2.0] * 5)
            ]
            videos.append(segment_video({"video_id": f"video-{video_index}", "frames": frames}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_videos_plateau_scatter.png"
            audit = render_all_video_plateau_scatter(
                videos[:1], path, eval_results=videos[1:],
                vx_seg_max_count=99, vz_seg_max_count=99,
                max_plateau_middle_th_vx=0.5,
                max_plateau_middle_th_vz=100.0,
            )
            expected = sum(
                len(video[key]["qualifying_plateaus"])
                for video in videos
                for key in ("vx_segmentation", "vz_segmentation")
            )
            self.assertEqual(audit["num_points"], expected)
            self.assertEqual(audit["num_train_videos"], 1)
            self.assertEqual(audit["num_eval_videos"], 1)
            self.assertEqual({row["split"] for row in audit["points"]}, {"train", "eval"})
            self.assertEqual(audit["vx_seg_max_count"], 99)
            self.assertEqual(audit["vz_seg_max_count"], 99)
            self.assertEqual(audit["max_plateau_middle_th_vx"], 0.5)
            self.assertEqual(audit["max_plateau_middle_th_vz"], 100.0)
            self.assertEqual(
                audit["num_disabled_points"],
                sum(
                    row["midpoint_n"] > (0.5 if row["axis"] == "vx" else 100.0)
                    for row in audit["points"]
                ),
            )
            self.assertEqual(
                audit["num_enabled_points"],
                sum(
                    row["midpoint_n"] <= (0.5 if row["axis"] == "vx" else 100.0)
                    for row in audit["points"]
                ),
            )
            self.assertTrue(all(
                row["enabled"] == (
                    row["midpoint_n"] <= (0.5 if row["axis"] == "vx" else 100.0)
                )
                for row in audit["points"]
            ))
            self.assertTrue(all(
                "plateau_middle_n_above_maximum" in row["disabled_reasons"]
                for row in audit["points"] if not row["enabled"]
            ))
            for axis in ("vx", "vz"):
                region = audit["confidence_regions"][axis]
                enabled_count = sum(
                    row["enabled"] and row["axis"] == axis and row["split"] == "train"
                    for row in audit["points"]
                )
                if enabled_count:
                    self.assertIsNotNone(region)
                    self.assertEqual(region["training_point_count"], enabled_count)
                    self.assertEqual(region["peak_confidence"], 1.0)
                else:
                    self.assertIsNone(region)
                metric = audit["evaluation_metrics"][axis]
                self.assertEqual(metric["metric"], "mean_eval_confidence")
                self.assertEqual(
                    metric["enabled_eval_points"],
                    sum(row["enabled"] and row["axis"] == axis and row["split"] == "eval" for row in audit["points"]),
                )
                if metric["value"] is not None:
                    self.assertGreaterEqual(metric["value"], 0.0)
                    self.assertLessEqual(metric["value"], 1.0)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
