import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import step7_train_eval_split, step7a_axis_threshold_segmentation, step7b_optimal_segmentation_selection

from src.exp_july.perception.ego_axis_threshold_visualization import render_eval_candidate_filter_comparisons, render_eval_signal_segmentation_chart

from src.exp_july.perception.ego_axis_threshold_segmentation import (
    _confidence_at,
    _confidence_surface,
    apply_semantic_candidate_confidence_correction,
    changed_label_confidence,
    confidence_weighted_consensus,
    filter_short_state_interruptions,
    finalize_enabled_consensus,
    materialize_enabled_candidates,
    merge_remaining_short_segments,
    render_all_video_plateau_scatter,
    render_segment_count_chart,
    render_train_optimal_n_scatter,
    segment_axis,
    segment_video,
)


class Step7AAxisThresholdSegmentationTests(unittest.TestCase):
    def test_step7b_merges_step7a_enabled_candidates_without_mutating_candidates(self):
        frames = [
            {"frame_index": index, "ego_vx": value, "ego_vz": value}
            for index, value in enumerate([-5.0] * 6 + [0.0] * 6 + [5.0] * 6)
        ]
        candidate_result = segment_video({"video_id": "demo", "frames": frames})
        candidate_result["data_split"] = "train"
        points = []
        for axis in ("vx", "vz"):
            for plateau in candidate_result[f"{axis}_segmentation"]["qualifying_plateaus"]:
                points.append({
                    "video_id": "demo", "axis": axis,
                    "plateau_id": int(plateau["plateau_id"]),
                    "midpoint_n": float(plateau["midpoint_n"]),
                    "enabled": True, "confidence": 0.9,
                    "disabled_reasons": [],
                })
        audit = {"points": points}
        materialize_enabled_candidates(candidate_result, audit)
        state = {
            "ego_axis_threshold_segmentation": [candidate_result],
            "ego_axis_threshold_segmentation_manifest": {"all_videos_plateau_scatter": audit},
            "ego_motion": [{"video_id": "demo", "frames": frames}],
            "step7_eval_video_ids": [],
            "step7_substeps": ["7a_axis_threshold_segmentation"],
        }
        config = {
            "consensus_min_segment_length_vx": 3,
            "consensus_min_segment_length_vz": 3,
            "visualization_max_eval_videos": 3,
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.exp_july.perception.pipeline.get_pipeline_output_root", return_value=Path(directory)), \
                patch("src.exp_july.perception.pipeline.driving_pipeline_config.get_step7a_axis_threshold_segmentation_cfg", return_value=config):
            output = step7b_optimal_segmentation_selection(state)
            manifest_exists = Path(output["ego_axis_consensus_segmentation_manifest_path"]).exists()
            final_path_exists = Path(output["ego_axis_final_segmentation"][0]["step7b_final_segmentation_path"]).exists()
            optimal_chart_exists = Path(output["ego_axis_consensus_segmentation_manifest"]["optimal_n_scatter"]["path"]).exists()
            shared_visual_root_exists = Path(output["ego_axis_consensus_segmentation_manifest"]["eval_visualization_output_root"]).is_dir()
        self.assertEqual(
            state["ego_axis_threshold_segmentation"][0]["final_segmentation"]["status"],
            "pending_step7b_consensus_merge",
        )
        final_result = output["ego_axis_final_segmentation"][0]
        self.assertEqual(final_result["final_segmentation"]["status"], "completed")
        self.assertEqual(final_result["final_segmentation"]["merge_step"], "7b")
        self.assertEqual(output["final_ego_symbols"][0]["source_step"], "7b_consensus_merge")
        self.assertEqual(final_result["vx_segmentation"]["optimal_n_selection"]["status"], "selected")
        self.assertEqual(final_result["vz_segmentation"]["optimal_n_selection"]["status"], "selected")
        self.assertTrue(optimal_chart_exists)
        self.assertTrue(shared_visual_root_exists)
        self.assertEqual(
            output["ego_axis_consensus_segmentation_manifest"]["eval_visualization_layout"],
            "single_shared_folder_video_id_filenames",
        )
        self.assertTrue(manifest_exists)
        self.assertTrue(final_path_exists)


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

    def test_changed_label_confidence_is_symmetric_and_reaches_zero_at_minimum_long_length(self):
        values = [changed_label_confidence(6, offset, 6) for offset in range(6)]
        self.assertEqual(values, [1.0, 0.5, 0.0, 0.0, 0.5, 1.0])
        self.assertEqual(values, list(reversed(values)))

    def test_optimal_n_heatmap_fits_train_and_overlays_eval(self):
        def result(video_id, split, vx_n, vz_n):
            def axis_selection(value, segments):
                return {
                    "optimal_n_selection": {
                        "status": "selected", "optimal_n": value,
                        "selected_segment_count": segments,
                        "selected_similarity": 0.9, "selected_candidate_id": 0,
                    }
                }
            return {
                "video_id": video_id, "data_split": split,
                "vx_segmentation": axis_selection(vx_n, 3),
                "vz_segmentation": axis_selection(vz_n, 2),
            }
        train = [result("train-a", "train", 10.0, 4.0), result("train-b", "train", 14.0, 6.0)]
        evaluation = [result("eval-a", "eval", 12.0, 5.0)]
        with tempfile.TemporaryDirectory() as directory:
            audit = render_train_optimal_n_scatter(
                train, evaluation, Path(directory) / "optimal.png",
                vx_seg_max_count=8, vz_seg_max_count=5,
                max_plateau_middle_th_vx=250.0, max_plateau_middle_th_vz=70.0,
            )
            self.assertTrue(Path(audit["path"]).exists())
        self.assertEqual(audit["num_train_optimal_points"], 4)
        self.assertEqual(audit["num_eval_optimal_points"], 2)
        self.assertFalse(audit["train_points_visible"])
        self.assertEqual(audit["visible_scatter_split"], "eval_only")
        self.assertEqual(audit["heatmap_style"]["colormap"], "viridis")
        self.assertEqual(audit["heatmap_style"]["opacity"], 1.0)
        self.assertEqual(audit["heatmap_style"]["contour_levels"], 25)
        self.assertEqual(audit["plot_limits_by_axis"]["vx"]["x_range_source"], "eval_optimal_n")
        self.assertLess(audit["plot_limits_by_axis"]["vx"]["x_min"], 12.0)
        self.assertGreater(audit["plot_limits_by_axis"]["vx"]["x_max"], 12.0)
        self.assertEqual(
            {row["split"] for row in audit["points"]}, {"train", "eval"},
        )
        eval_points = [row for row in audit["points"] if row["split"] == "eval"]
        self.assertTrue(all(row["train_density_confidence"] is not None for row in eval_points))

    def test_step7b_semantic_correction_penalizes_both_opposite_segments(self):
        def candidate(candidate_index, states):
            segments = []
            frame_labels = []
            start = 0
            for segment_id, (state, duration) in enumerate(states):
                end = start + duration - 1
                segments.append({
                    "segment_id": segment_id, "state": state,
                    "start_frame": start, "end_frame": end,
                    "duration_frames": duration,
                })
                frame_labels.extend({
                    "frame_index": frame_index, "label": state, "confidence": 1.0,
                } for frame_index in range(start, end + 1))
                start = end + 1
            return {
                "candidate_index": candidate_index, "threshold": float(candidate_index + 1),
                "candidate_confidence": 1.0, "segments": segments,
                "frame_labels": frame_labels,
            }

        violating = candidate(0, [("forward", 4), ("backward", 4)])
        compliant = candidate(1, [("static", 8)])
        result = {
            "vx_segmentation": {
                "labels": {"negative": "right", "center": "straight", "positive": "left"},
                "enabled_segmentation_candidates": [],
                "candidate_selection_summary": {},
            },
            "vz_segmentation": {
                "labels": {"negative": "backward", "center": "static", "positive": "forward"},
                "enabled_segmentation_candidates": [violating, compliant],
                "candidate_selection_summary": {
                    "num_qualifying_candidates": 2, "num_disabled_candidates": 0,
                },
            },
        }
        summary = apply_semantic_candidate_confidence_correction(
            result, opposite_transition_penalty=0.75,
        )
        self.assertEqual(summary["num_violations"], 1)
        self.assertEqual(summary["num_penalized_candidates"], 1)
        self.assertTrue(all(
            row["semantic_corrected_confidence"] == 0.25
            for row in violating["frame_labels"]
        ))
        self.assertTrue(all(
            row["semantic_confidence_multiplier"] == 0.25
            for row in violating["segments"]
        ))
        self.assertTrue(all(
            row["semantic_corrected_confidence"] == 1.0
            for row in compliant["frame_labels"]
        ))
        finalize_enabled_consensus(
            result, None, vx_minimum_segment_length=2, vz_minimum_segment_length=2,
        )
        final_states = [
            row["state"]
            for row in result["vz_segmentation"]["final_segmentation"]["frames"]
        ]
        self.assertEqual(final_states, ["static"] * 8)

    def test_consensus_dp_returns_one_sequence_with_frame_diagnostics(self):
        candidates = []
        for candidate_index in range(3):
            frame_labels = []
            for frame_index in range(7):
                label = "forward"
                if frame_index == 3 and candidate_index in (1, 2):
                    label = "backward"
                frame_labels.append({
                    "frame_index": frame_index,
                    "label": label,
                    "confidence": 1.0,
                })
            candidates.append({
                "candidate_index": candidate_index,
                "threshold": float(candidate_index + 1),
                "frame_labels": frame_labels,
            })
        final = confidence_weighted_consensus(
            candidates, ("backward", "static", "forward"), 3,
        )
        self.assertTrue(final["authoritative"])
        self.assertEqual(final["num_candidates"], 3)
        self.assertEqual(len(final["frames"]), 7)
        self.assertEqual([row["state"] for row in final["frames"]], ["forward"] * 7)
        contested = final["frames"][3]
        self.assertEqual(contested["local_evidence_winner"], "backward")
        self.assertTrue(contested["dp_overrode_local_winner"])
        self.assertAlmostEqual(contested["confidence"], 1.0 / 3.0)
        self.assertAlmostEqual(contested["consensus"], 1.0 / 3.0)
        self.assertLess(contested["margin"], 0.0)
        self.assertAlmostEqual(contested["candidate_disagreement"], 1.0 / 3.0)
        self.assertTrue(all(
            row["duration_frames"] >= 3 for row in final["segments"]
        ))

    def test_enabled_plateau_candidates_only_feed_final_consensus(self):
        frames = [
            {"frame_index": index, "ego_vx": value, "ego_vz": value}
            for index, value in enumerate([-5.0] * 6 + [0.0] * 6 + [5.0] * 6)
        ]
        result = segment_video(
            {"video_id": "enabled-only", "frames": frames},
            vx_noise_tolerance_frames=2, vz_noise_tolerance_frames=2,
            plateau_min_n_values=3,
            vx_consensus_min_segment_length=3,
            vz_consensus_min_segment_length=3,
        )
        self.assertEqual(
            result["final_segmentation"]["vx"]["status"],
            "pending_enabled_candidate_audit",
        )
        points = []
        enabled_ids = {}
        for axis in ("vx", "vz"):
            plateaus = result[f"{axis}_segmentation"]["qualifying_plateaus"]
            self.assertTrue(plateaus)
            enabled = plateaus[:1]
            enabled_ids[axis] = [int(row["plateau_id"]) for row in enabled]
            for plateau in plateaus:
                points.append({
                    "video_id": "enabled-only",
                    "axis": axis,
                    "plateau_id": int(plateau["plateau_id"]),
                    "midpoint_n": float(plateau["midpoint_n"]),
                    "enabled": plateau in enabled,
                    "confidence": 0.8 if plateau in enabled else None,
                })
        finalize_enabled_consensus(
            result, {"points": points},
            vx_minimum_segment_length=3,
            vz_minimum_segment_length=3,
        )
        for axis in ("vx", "vz"):
            final = result["final_segmentation"][axis]
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["candidate_scope"], "enabled_qualifying_plateau_middle_candidates")
            self.assertEqual(final["enabled_candidate_ids"], enabled_ids[axis])
            self.assertEqual(final["num_candidates"], 1)
            self.assertTrue(final["disabled_candidates_excluded"])
            self.assertEqual(len(final["frames"]), len(frames))
            self.assertTrue(all(0.0 <= row["confidence"] <= 1.0 for row in final["frames"]))

    def test_no_enabled_plateau_produces_explicit_unavailable_result(self):
        frames = [
            {"frame_index": index, "ego_vx": value, "ego_vz": value}
            for index, value in enumerate([-5.0] * 6 + [0.0] * 6 + [5.0] * 6)
        ]
        result = segment_video({"video_id": "none-enabled", "frames": frames})
        finalize_enabled_consensus(result, {"points": []})
        for axis in ("vx", "vz"):
            final = result["final_segmentation"][axis]
            self.assertEqual(final["status"], "unavailable_no_enabled_candidates")
            self.assertFalse(final["authoritative"])
            self.assertEqual(final["frames"], [])
            self.assertEqual(final["segments"], [])


    def test_candidate_frame_labels_receive_confidence_after_short_merge(self):
        frames = [
            {"frame_index": index, "ego_vz_smoothed": value}
            for index, value in enumerate([10.0] * 8 + [-10.0] * 5 + [10.0] * 8)
        ]
        result = segment_axis(
            frames, "vz", ("backward", "static", "forward"),
            noise_tolerance_frames=5,
        )
        candidate = result["threshold_candidates"][0]
        labels = candidate["frame_labels"]
        self.assertEqual(len(labels), len(frames))
        self.assertTrue(all(row["confidence"] == 1.0 for row in labels[:8] + labels[13:]))
        changed = labels[8:13]
        self.assertTrue(all(row["label_changed"] for row in changed))
        self.assertTrue(all(row["original_label_confidence"] == 1.0 for row in labels))
        self.assertTrue(all(row["filtered_label_confidence"] == row["confidence"] for row in labels))
        self.assertEqual([row["original_label"] for row in changed], ["backward"] * 5)
        self.assertEqual([row["label"] for row in changed], ["forward"] * 5)
        confidence = [row["confidence"] for row in changed]
        self.assertEqual(confidence, list(reversed(confidence)))
        self.assertAlmostEqual(confidence[0], 1.0)
        self.assertAlmostEqual(confidence[2], 1.0 - 5.0 / 6.0)
        self.assertEqual(result["frame_label_confidence"]["minimum_long_segment_length"], 6)

    def test_noise_filter_merges_single_short_state_between_long_matching_states(self):
        segments = [
            {"state": "forward", "start_frame": 0, "end_frame": 9, "duration_frames": 10},
            {"state": "static", "start_frame": 10, "end_frame": 13, "duration_frames": 4},
            {"state": "forward", "start_frame": 14, "end_frame": 23, "duration_frames": 10},
        ]
        filtered = filter_short_state_interruptions(segments, tolerance_frames=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["state"], "forward")
        self.assertEqual(filtered[0]["duration_frames"], 24)
        self.assertEqual(filtered[0]["absorbed_states"], ["static"])

    def test_noise_filter_merges_multi_state_interruption_for_either_axis(self):
        segments = [
            {"state": "left", "start_frame": 0, "end_frame": 7, "duration_frames": 8},
            {"state": "straight", "start_frame": 8, "end_frame": 8, "duration_frames": 1},
            {"state": "right", "start_frame": 9, "end_frame": 9, "duration_frames": 1},
            {"state": "straight", "start_frame": 10, "end_frame": 10, "duration_frames": 1},
            {"state": "left", "start_frame": 11, "end_frame": 18, "duration_frames": 8},
        ]
        filtered = filter_short_state_interruptions(segments, tolerance_frames=5)
        self.assertEqual([row["state"] for row in filtered], ["left"])
        self.assertEqual(filtered[0]["absorbed_interruption_frames"], 3)
        self.assertEqual(filtered[0]["absorbed_states"], ["straight", "right", "straight"])

    def test_noise_filter_merges_complex_sequence_of_individually_short_states(self):
        states = [
            ("forward", 20),
            ("backward", 2),
            ("static", 2),
            ("forward", 1),
            ("backward", 2),
            ("forward", 25),
        ]
        segments = []
        start = 0
        for state, duration in states:
            segments.append({"state": state, "start_frame": start, "end_frame": start + duration - 1, "duration_frames": duration})
            start += duration
        filtered = filter_short_state_interruptions(
            segments, tolerance_frames=5,
            bridge_total_max_frames=15, anchor_min_frames=8,
            bridge_max_segments=5, bridge_max_anchor_ratio=0.75,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["state"], "forward")
        self.assertEqual(filtered[0]["duration_frames"], 52)
        self.assertEqual(filtered[0]["absorbed_segment_count"], 4)
        self.assertEqual(filtered[0]["absorbed_states"], ["backward", "static", "forward", "backward"])

    def test_noise_filter_rejects_bridge_with_excessive_anchor_ratio(self):
        segments = [
            {"state": "forward", "start_frame": 0, "end_frame": 7, "duration_frames": 8},
            {"state": "static", "start_frame": 8, "end_frame": 13, "duration_frames": 6},
            {"state": "forward", "start_frame": 14, "end_frame": 21, "duration_frames": 8},
        ]
        filtered = filter_short_state_interruptions(
            segments, tolerance_frames=6,
            bridge_total_max_frames=15, anchor_min_frames=8,
            bridge_max_segments=5, bridge_max_anchor_ratio=0.5,
        )
        self.assertEqual(len(filtered), 3)

    def test_residual_short_island_uses_mean_signal_to_split_between_long_neighbors(self):
        segments = [
            {"state": "forward", "start_frame": 0, "end_frame": 9, "duration_frames": 10, "mean_signal": 10.0},
            {"state": "static", "start_frame": 10, "end_frame": 11, "duration_frames": 2, "mean_signal": 8.0},
            {"state": "static", "start_frame": 12, "end_frame": 13, "duration_frames": 2, "mean_signal": -9.0},
            {"state": "backward", "start_frame": 14, "end_frame": 23, "duration_frames": 10, "mean_signal": -10.0},
        ]
        filtered = merge_remaining_short_segments(segments, tolerance_frames=5)
        self.assertEqual([row["state"] for row in filtered], ["forward", "backward"])
        self.assertEqual([row["duration_frames"] for row in filtered], [12, 12])
        self.assertTrue(all(row["duration_frames"] > 5 for row in filtered))
        self.assertEqual(filtered[0]["residual_short_assignments"][0]["assigned_state"], "forward")
        self.assertEqual(filtered[1]["residual_short_assignments"][0]["assigned_state"], "backward")

    def test_residual_edge_short_island_attaches_to_only_long_neighbor(self):
        segments = [
            {"state": "static", "start_frame": 0, "end_frame": 2, "duration_frames": 3, "mean_signal": 0.0},
            {"state": "forward", "start_frame": 3, "end_frame": 12, "duration_frames": 10, "mean_signal": 8.0},
        ]
        filtered = merge_remaining_short_segments(segments, tolerance_frames=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["state"], "forward")
        self.assertEqual(filtered[0]["duration_frames"], 13)

    def test_noise_filter_preserves_interruptions_above_tolerance(self):
        segments = [
            {"state": "backward", "start_frame": 0, "end_frame": 7, "duration_frames": 8},
            {"state": "static", "start_frame": 8, "end_frame": 13, "duration_frames": 6},
            {"state": "backward", "start_frame": 14, "end_frame": 21, "duration_frames": 8},
        ]
        self.assertEqual(len(filter_short_state_interruptions(segments, tolerance_frames=5)), 3)

    def test_pipeline_writes_per_video_outputs_under_train_and_eval_folders(self):
        train_frames = [{"frame_index": index, "ego_vx": float(index % 3 - 1), "ego_vz": float(index % 3 - 1)} for index in range(6)]
        eval_frames = [{"frame_index": index, "ego_vx": float(1 - index % 3), "ego_vz": float(1 - index % 3)} for index in range(6)]
        ego_motion = [
            {"video_id": "train-video", "frames": train_frames},
            {"video_id": "eval-video", "frames": eval_frames},
        ]
        config = {
            "vx_seg_max_count": 8, "vz_seg_max_count": 5,
            "max_plateau_middle_th_vx": 250.0, "max_plateau_middle_th_vz": 70.0,
            "plateau_min_n_values": 3,
            "noise_tolerance_frames_vx": 5, "noise_tolerance_frames_vz": 5,
            "bridge_total_max_frames_vx": 15, "bridge_total_max_frames_vz": 15,
            "anchor_min_frames_vx": 8, "anchor_min_frames_vz": 8,
            "bridge_max_segments_vx": 5, "bridge_max_segments_vz": 5,
            "bridge_max_anchor_ratio_vx": 0.75, "bridge_max_anchor_ratio_vz": 0.75,
            "filter_comparison_max_candidates": 2,
        }

        def touch_chart(_result, path, **_kwargs):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"chart"); return str(path)

        def scatter(_train, path, **_kwargs):
            path = Path(path); path.write_bytes(b"scatter"); return {"path": str(path), "points": []}

        def visual(_result, _ego, _audit, path, **_kwargs):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"mp4"); return {"status": "rendered", "path": str(path)}

        def signal_chart(_result, _audit, path, **_kwargs):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"png"); return {"status": "rendered", "path": str(path)}

        def filter_comparisons(_result, output_root, max_candidates=20, **_kwargs):
            charts = []
            for axis in ("vx", "vz"):
                for index in range(max_candidates):
                    path = Path(output_root) / axis / f"candidate_{index:02d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"png")
                    charts.append({"status": "rendered", "axis": axis, "path": str(path)})
            return {"status": "rendered", "num_charts": len(charts), "charts": charts}

        with tempfile.TemporaryDirectory() as directory, \
                patch("src.exp_july.perception.pipeline.get_pipeline_output_root", return_value=Path(directory)), \
                patch("src.exp_july.perception.pipeline.step7_ego_motion", return_value={"ego_motion": ego_motion}), \
                patch("src.exp_july.perception.pipeline.driving_pipeline_config.get_step7a_axis_threshold_segmentation_cfg", return_value=config), \
                patch("src.exp_july.perception.ego_axis_threshold_segmentation.render_segment_count_chart", side_effect=touch_chart), \
                patch("src.exp_july.perception.ego_axis_threshold_segmentation.render_all_video_plateau_scatter", side_effect=scatter), \
                patch("src.exp_july.perception.ego_axis_threshold_visualization.render_axis_segmentation_mp4", side_effect=visual), \
                patch("src.exp_july.perception.ego_axis_threshold_visualization.render_eval_signal_segmentation_chart", side_effect=signal_chart), \
                patch("src.exp_july.perception.ego_axis_threshold_visualization.render_eval_candidate_filter_comparisons", side_effect=filter_comparisons):
            output = step7a_axis_threshold_segmentation({
                "step7_train_video_ids": ["train-video"],
                "step7_eval_video_ids": ["eval-video"],
            }, render_candidate_filter_comparisons=True)
            root = Path(output["ego_axis_threshold_segmentation_output_root"])
            self.assertTrue((root / "train" / "train-video" / "axis_threshold_segmentation.json").exists())
            self.assertFalse((root / "train" / "train-video" / "axis_threshold_segment_counts.png").exists())
            self.assertTrue((root / "eval" / "eval-video" / "axis_threshold_segmentation.json").exists())
            self.assertTrue((root / "eval" / "eval-video" / "axis_segmentation_visualization.mp4").exists())
            self.assertTrue((root / "eval" / "eval-video" / "axis_signal_segmentation.png").exists())
            self.assertTrue((root / "eval" / "eval-video" / "candidate_filter_comparisons" / "vx" / "candidate_00.png").exists())
            self.assertTrue((root / "eval" / "eval-video" / "candidate_filter_comparisons" / "vz" / "candidate_00.png").exists())
            self.assertFalse((root / "eval" / "train-video").exists())
            self.assertEqual(output["ego_axis_threshold_segmentation_manifest"]["num_train_videos"], 1)
            self.assertEqual(output["ego_axis_threshold_segmentation_manifest"]["num_eval_videos"], 1)

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

    def test_threshold_candidates_preserve_raw_and_filtered_segment_counts(self):
        frames = [
            {"frame_index": index, "ego_vz_smoothed": value}
            for index, value in enumerate([-3.0] * 10 + [0.0] * 3 + [-3.0] * 10)
        ]
        result = segment_axis(
            frames, "vz", ("backward", "static", "forward"),
            noise_tolerance_frames=5,
        )
        affected = [
            row for row in result["threshold_candidates"]
            if row["raw_segment_count"] > row["segment_count"]
        ]
        self.assertTrue(affected)
        self.assertTrue(all(row["raw_segment_count"] == 3 for row in affected))
        self.assertTrue(all(row["segment_count"] == 1 for row in affected))
        self.assertTrue(all("raw_segment_count_min" in row for row in result["all_plateaus"]))
        self.assertTrue(all("raw_segment_count_max" in row for row in result["all_plateaus"]))

    def test_keeps_every_long_plateau_and_uses_its_middle_n(self):
        frames = [
            {"frame_index": index, "ego_vz_smoothed": value}
            for index, value in enumerate([-3.0] * 8 + [0.0] * 8 + [3.0] * 8)
        ]
        result = segment_axis(frames, "vz", ("backward", "static", "forward"))
        self.assertTrue(result["qualifying_plateaus"])
        self.assertNotIn("optimal_threshold", result)
        self.assertNotIn("selected_plateau", result)
        for plateau in result["qualifying_plateaus"]:
            self.assertGreaterEqual(plateau["num_n_values"], 3)
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
            [(-2.0, -3.0)] * 8
            + [(0.0, 0.0)] * 8
            + [(2.0, 3.0)] * 8
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


    def test_renders_smallest_candidate_filter_comparisons_for_both_axes(self):
        frames = [
            {"frame_index": index, "ego_vx": vx, "ego_vz": vz}
            for index, (vx, vz) in enumerate(
                [(-3.0, 3.0)] * 8 + [(0.0, 0.0)] * 3 + [(-3.0, 3.0)] * 8
            )
        ]
        result = segment_video({"video_id": "filter-demo", "frames": frames})
        with tempfile.TemporaryDirectory() as directory:
            output = render_eval_candidate_filter_comparisons(
                result, Path(directory), max_candidates=2,
            )
            self.assertEqual(output["num_charts"], 4)
            self.assertEqual(output["max_candidates_per_axis"], 2)
            self.assertEqual([row["axis"] for row in output["charts"]], ["vx", "vx", "vz", "vz"])
            for row in output["charts"]:
                path = Path(row["path"])
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)
                self.assertGreaterEqual(row["raw_segment_count"], row["filtered_segment_count"])
                self.assertTrue(all(segment["length_class"] in {"short", "long"} for segment in row["raw_segments"]))
                self.assertTrue(all(segment["length_class"] in {"short", "long"} for segment in row["filtered_segments"]))
                self.assertEqual(row["short_segment_definition"], "duration_frames <= 5")
                self.assertEqual(row["long_segment_definition"], "duration_frames > 5")
                self.assertEqual(row["raw_frame_confidence_min"], 1.0)
                self.assertEqual(row["raw_frame_confidence_max"], 1.0)
                self.assertGreaterEqual(row["filtered_frame_confidence_max"], row["filtered_frame_confidence_min"])
                self.assertGreaterEqual(row["filtered_frame_confidence_min"], 0.0)
                self.assertLessEqual(row["filtered_frame_confidence_max"], 1.0)
            self.assertIn("short", {segment["length_class"] for segment in output["charts"][0]["raw_segments"]})
            self.assertIn("long", {segment["length_class"] for segment in output["charts"][0]["raw_segments"]})
            self.assertEqual(output["layout"], "4x1_before_after_segmentation_and_confidence")
            for axis in ("vx", "vz"):
                thresholds = [row["threshold_n"] for row in output["charts"] if row["axis"] == axis]
                expected = [row["threshold"] for row in result[f"{axis}_segmentation"]["threshold_candidates"][:2]]
                self.assertEqual(thresholds, expected)

    def test_signal_chart_renders_enabled_and_disabled_candidates(self):
        frames = [
            {"frame_index": index, "ego_vx": float(index % 3 - 1), "ego_vz": float(index % 3 - 1)}
            for index in range(12)
        ]
        result = {
            "video_id": "status-demo", "frames": frames,
            "vx_segmentation": {"qualifying_plateaus": [
                {"plateau_id": 1, "midpoint_n": 1.0, "segments": []},
                {"plateau_id": 2, "midpoint_n": 2.0, "segments": []},
            ]},
            "vz_segmentation": {"qualifying_plateaus": [
                {"plateau_id": 3, "midpoint_n": 3.0, "segments": []},
            ]},
        }
        audit = {"points": [
            {"video_id": "status-demo", "axis": "vx", "plateau_id": 1, "enabled": True, "confidence": 0.8, "disabled_reasons": []},
            {"video_id": "status-demo", "axis": "vx", "plateau_id": 2, "enabled": False, "confidence": None, "disabled_reasons": ["segment_count_above_seg_max_count"]},
            {"video_id": "status-demo", "axis": "vz", "plateau_id": 3, "enabled": False, "confidence": None, "disabled_reasons": ["plateau_middle_n_above_maximum"]},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "axis_signal_segmentation.png"
            output = render_eval_signal_segmentation_chart(result, audit, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(len(output["vx_enabled_candidates"]), 1)
            self.assertEqual(len(output["vx_disabled_candidates"]), 1)
            self.assertEqual(len(output["vz_disabled_candidates"]), 1)
            self.assertEqual(output["vx_disabled_candidates"][0]["activation_status"], "DISABLED")
            self.assertEqual(output["layout"], "k_by_2_all_qualifying_threshold_segmentations")

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
                for index, value in enumerate([-2.0] * 8 + [0.0] * 8 + [2.0] * 8)
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
                audit["plot_limits_by_axis"],
                {
                    "vx": {"x_min": 0.0, "x_max": 0.6, "y_min": 0.0, "y_max": 118.8},
                    "vz": {"x_min": 0.0, "x_max": 120.0, "y_min": 0.0, "y_max": 118.8},
                },
            )
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
                    self.assertEqual(region["bounds"], audit["plot_limits_by_axis"][axis])
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
