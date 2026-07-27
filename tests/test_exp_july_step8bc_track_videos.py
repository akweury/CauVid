import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.trajectory_pattern_closed_loop import (
    PATTERNS,
    RESIDUALS,
)
from src.exp_july.perception.trajectory_pattern_visualization import (
    _appearance_marker_offset,
    _bbox_difference_metrics,
    _build_track_signal_versions_panel,
    _cue_visual_state,
    _ego_speed_series,
    _object_track_velocity_series,
    _signal_values,
    _track_motion_series,
    build_step8bc_track_video_payload,
    render_step8bc_track_videos,
    render_trajectory_pattern_visualizations,
    select_deterministic_track_records,
)


def _record(video_id, track_id):
    signal_evidence = {
        "track_id": track_id,
        "primary_label": "car",
        "observable_cues": {
            "leftness": 0.0,
            "rightness": 0.84,
            "approach": 0.86,
            "recede": 0.0,
            "acceleration": 0.52,
            "deceleration": 0.0,
        },
    }
    pattern_candidates = []
    final_pattern_candidates = []
    for pattern_index, pattern_id in enumerate(PATTERNS):
        before = {
            residual: float(1000 * track_id + 100 * pattern_index + residual_index)
            for residual_index, residual in enumerate(RESIDUALS)
        }
        after = {
            residual: value + 0.25
            for residual, value in before.items()
        }
        pattern_candidates.append(
            {"pattern_id": pattern_id, "residual_vector": before}
        )
        final_pattern_candidates.append(
            {"pattern_id": pattern_id, "residual_vector": after}
        )
    return {
        "video_id": video_id,
        "track_id": track_id,
        "symbolic_track": {
            "object_class": "car",
            "trajectory_statistics": {
                "max_frame_gap": 3,
                "has_motion_ratio": 0.8125,
            },
            "position": {
                "start": [0.0, 0.0, 12.0],
                "end": [1.0, 0.0, 10.0],
                "path_length_xz": 2.25,
            },
            "bbox_size": {"mean": 900.0, "max_abs_step": 20.0},
            "relative_motion": {"mean": 1.25, "max": 2.5},
            "persistence": 0.875,
            "confidence": 0.925,
            "source_evidence_type": "uncertain_signal_evidence",
            "source_signal_evidence": signal_evidence,
            "observable_cues": signal_evidence["observable_cues"],
        },
        "pattern_candidates": pattern_candidates,
        "final_pattern_candidates": final_pattern_candidates,
        "candidate_repairs": [
            {
                "candidate_id": "approaching:motion_recomputation",
                "pattern_id": "approaching",
                "residual_vector_before": pattern_candidates[3][
                    "residual_vector"
                ],
                "residual_vector_after": final_pattern_candidates[3][
                    "residual_vector"
                ],
                "residual_improvement": 0.375,
            },
            {
                "candidate_id": "stationary:kalman_smoothing",
                "pattern_id": "stationary",
                "residual_vector_before": pattern_candidates[0][
                    "residual_vector"
                ],
                "residual_vector_after": final_pattern_candidates[0][
                    "residual_vector"
                ],
                "residual_improvement": 0.625,
            },
        ],
        "validated_pattern": "approaching",
        "final_pattern": "approaching",
        "final_validation_status": "valid",
        "repair_applied": False,
        "resolution_status": "validated_no_repair",
        "trajectory_cohort_id": "persistent_vehicle",
        "activated_rule": {
            "rule_id": "persistent_vehicle",
            "description": "Persistent vehicle trajectories",
        },
        "cohort_static_metadata": {
            "category": "car",
            "track_length_bucket": "medium",
        },
        "cohort_statistical_summary": {
            "track_count": 12,
            "systematic_anomalies": [],
        },
        "cohort_operator_plan": {
            "operator": "no_repair",
            "llm_requested_operator": "outlier_removal",
            "calibrated_parameters": {},
            "calibration": {
                "promotion_decision": "no_repair_required",
                "selected_measurement": {
                    "sample_count": 0,
                    "success_rate": 0.0,
                    "mean_issue_cost_improvement": 0.0,
                },
            },
        },
    }


class Step8BCTrackVideoTests(unittest.TestCase):
    def test_ego_speed_series_prefers_refined_values_and_aligns_frames(self):
        ego_video = {
            "frames": [
                {
                    "frame_index": 0,
                    "ego_vx": 1.0,
                    "ego_vz": 2.0,
                },
                {
                    "frame_index": 2,
                    "refined_ego_vx": 3.0,
                    "ego_vx_smoothed": 2.5,
                    "refined_ego_vz": -1.0,
                },
            ]
        }
        series = _ego_speed_series(ego_video, [0, 1, 2])
        self.assertEqual(series["vx"], [1.0, None, 3.0])
        self.assertEqual(series["vz"], [2.0, None, -1.0])

    def test_third_panel_persists_original_and_optional_repaired_charts(self):
        import cv2
        import numpy as np

        original_track = {
            10: {"obj_vx": 0.2, "obj_vz": 1.1},
            12: {"obj_vx": 0.4, "obj_vz": 1.3},
            15: {"motion": {"obj_vx": 0.3, "obj_vz": 1.2}},
        }
        repaired_track = {
            10: {"obj_vx": 0.2, "obj_vz": 1.0},
            12: {"obj_vx": 0.3, "obj_vz": 1.1},
            15: {"obj_vx": 0.25, "obj_vz": 1.05},
        }
        indices, values = _object_track_velocity_series(original_track, "x")
        self.assertEqual(indices, [10, 12, 15])
        self.assertEqual(values, [0.2, 0.4, 0.3])
        self.assertEqual(_appearance_marker_offset(indices, 9), 0)
        self.assertEqual(_appearance_marker_offset(indices, 11), 0)
        self.assertEqual(_appearance_marker_offset(indices, 12), 1)
        self.assertEqual(_appearance_marker_offset(indices, 15), 2)

        with patch(
            "src.exp_july.perception.trajectory_pattern_visualization._text"
        ) as draw_text:
            _build_track_signal_versions_panel(
                cv2,
                np,
                frame_index=12,
                original_track=original_track,
                repaired_track=repaired_track,
                repair_applied=True,
                width=560,
                height=1440,
            )
            rendered = [str(call.args[2]) for call in draw_text.call_args_list]
        self.assertEqual(
            rendered,
            [
                "TRACK SIGNAL VERSIONS",
                "ORIGINAL OBJ VX [8A]",
                "ORIGINAL OBJ VZ [8A]",
                "REPAIRED OBJ VX [8D]",
                "REPAIRED OBJ VZ [8D]",
            ],
        )

        with patch(
            "src.exp_july.perception.trajectory_pattern_visualization._text"
        ) as draw_text:
            _build_track_signal_versions_panel(
                cv2,
                np,
                frame_index=9,
                original_track=original_track,
                repaired_track=repaired_track,
                repair_applied=False,
                width=560,
                height=1440,
            )
            rendered = [str(call.args[2]) for call in draw_text.call_args_list]
        self.assertEqual(
            rendered,
            [
                "TRACK SIGNAL VERSIONS",
                "ORIGINAL OBJ VX [8A]",
                "ORIGINAL OBJ VZ [8A]",
                "REPAIRED OBJ VX [8D]",
                "NO REPAIR",
                "REPAIRED OBJ VZ [8D]",
                "NO REPAIR",
            ],
        )

    def test_track_motion_series_aligns_object_and_relative_velocity(self):
        track = {
            0: {
                "obj_vx": 1.25,
                "obj_vz": -0.5,
                "rel_vx": 0.75,
                "rel_vz": -1.5,
            },
            2: {
                "motion": {
                    "obj_vx": 2.0,
                    "obj_vz": 3.0,
                    "rel_vx": 1.0,
                    "rel_vz": 1.5,
                }
            },
        }
        series = _track_motion_series(track, [0, 1, 2])
        self.assertEqual(series["obj_vx"], [1.25, None, 2.0])
        self.assertEqual(series["obj_vz"], [-0.5, None, 3.0])
        self.assertEqual(series["rel_vx"], [0.75, None, 1.0])
        self.assertEqual(series["rel_vz"], [-1.5, None, 1.5])

    def test_cues_activate_only_when_object_is_observed_in_current_frame(self):
        text, color, thickness, active = _cue_visual_state(
            "approach", 0.8, True
        )
        self.assertTrue(active)
        self.assertEqual(color, (70, 220, 100))
        self.assertEqual(thickness, 2)
        self.assertEqual(text, "approach=0.80")

        text, color, thickness, active = _cue_visual_state(
            "approach", 0.8, False
        )
        self.assertFalse(active)
        self.assertEqual(color, (145, 152, 163))
        self.assertEqual(thickness, 1)
        self.assertEqual(text, "approach=0.80")

    def test_original_repaired_difference_metrics(self):
        original = {
            "position_3d": [1.0, 0.0, 10.0],
            "rel_vx": 2.0,
            "rel_vz": -1.0,
            "rel_speed": 2.25,
            "bbox": [10.0, 20.0, 30.0, 40.0],
        }
        repaired = {
            "position_3d": [2.0, 0.0, 8.0],
            "rel_vx": 1.0,
            "rel_vz": -3.0,
            "rel_speed": 3.2,
            "bbox": [12.0, 20.0, 32.0, 40.0],
        }
        self.assertEqual(_signal_values(original), (1.0, 10.0, 2.0, -1.0, 2.25))
        difference = _bbox_difference_metrics(original, repaired)
        self.assertAlmostEqual(difference["center_shift_px"], 2.0)
        self.assertAlmostEqual(difference["iou"], 18.0 / 22.0)

    def test_step8h_saves_only_mp4_and_pdf_artifacts(self):
        record = _record("scene", 7)
        state = {
            "trajectory_pattern_records": [record],
            "trajectory_pattern_statistics_promotion": {
                "decision": "reject",
                "reason": "validation_regression",
            },
        }
        empty_video_manifest = {
            "num_selected_tracks": 0,
            "num_rendered_videos": 0,
            "num_skipped_videos": 0,
            "rendered": [],
            "skipped": [],
            "selections": [],
            "manifest_path": "",
        }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception.trajectory_pattern_visualization."
            "render_step8bc_track_videos",
            return_value=empty_video_manifest,
        ):
            root = Path(tmp)
            (root / "stale.json").write_text("{}", encoding="utf-8")
            (root / "stale.html").write_text("stale", encoding="utf-8")
            result = render_trajectory_pattern_visualizations(state, root)
            pdf_paths = sorted((root / "statistics_pdfs").glob("*.pdf"))
            self.assertEqual(len(pdf_paths), 3)
            self.assertTrue(all(path.stat().st_size > 0 for path in pdf_paths))
            files = [path for path in root.rglob("*") if path.is_file()]
            self.assertTrue(files)
            self.assertEqual({path.suffix.lower() for path in files}, {".pdf"})
            self.assertEqual(
                len(result["trajectory_pattern_statistical_pdf_reports"]),
                3,
            )
            self.assertEqual(result["trajectory_pattern_visualizations"], [])
            self.assertEqual(
                result["trajectory_pattern_statistical_summary_path"], ""
            )

    def test_selection_is_stable_order_independent_and_capped_globally(self):
        records = (
            [_record("scene_b", track_id) for track_id in range(17)]
            + [_record("scene_a", track_id) for track_id in range(6)]
            + [_record("scene_b", 3)]
        )

        selected = select_deterministic_track_records(
            records, max_tracks_per_video=10
        )
        selected_again = select_deterministic_track_records(
            list(reversed(records)), max_tracks_per_video=10
        )
        selected_with_default = select_deterministic_track_records(records)
        selected_with_oversized_limit = select_deterministic_track_records(
            records, max_tracks_per_video=100
        )

        selected_keys = [
            (row["video_id"], row["track_id"]) for row in selected
        ]
        selected_again_keys = [
            (row["video_id"], row["track_id"]) for row in selected_again
        ]
        self.assertEqual(selected_keys, selected_again_keys)
        self.assertEqual(
            selected_keys,
            [
                (row["video_id"], row["track_id"])
                for row in selected_with_default
            ],
        )
        self.assertEqual(
            selected_keys,
            [
                (row["video_id"], row["track_id"])
                for row in selected_with_oversized_limit
            ],
        )
        self.assertEqual(len(selected_keys), len(set(selected_keys)))
        self.assertEqual(len(selected_keys), 5)

    def test_payload_keeps_8b_signal_evidence_and_every_8c_residual_distance(self):
        record = _record("scene", 7)

        payload = build_step8bc_track_video_payload(record)

        self.assertEqual(payload["video_id"], "scene")
        self.assertEqual(payload["track_id"], 7)
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(
            payload["step8c"]["trajectory_cohort_id"],
            "persistent_vehicle",
        )
        self.assertEqual(
            payload["step8c"]["cohort_operator_plan"]["operator"],
            "no_repair",
        )
        serialized = json.dumps(payload, sort_keys=True)
        signal_evidence = record["symbolic_track"]["source_signal_evidence"]
        self.assertEqual(
            payload["step8b_signal_evidence"], signal_evidence
        )
        signal_serialized = json.dumps(
            payload["step8b_signal_evidence"], sort_keys=True
        )
        for required_8b_field in (
            "leftness",
            "rightness",
            "approach",
            "recede",
            "acceleration",
            "deceleration",
        ):
            self.assertIn(required_8b_field, serialized)
        for forbidden_8b_field in (
            "validation_status",
            "source_decision",
            "fact_decision_status",
        ):
            self.assertNotIn(forbidden_8b_field, signal_serialized)
        for candidate_key in ("pattern_candidates", "final_pattern_candidates"):
            for candidate in record[candidate_key]:
                self.assertIn(candidate["pattern_id"], serialized)
                for residual_name, residual_value in candidate[
                    "residual_vector"
                ].items():
                    self.assertIn(residual_name, serialized)
                    self.assertIn(json.dumps(residual_value), serialized)
        for repair in record["candidate_repairs"]:
            self.assertIn(repair["candidate_id"], serialized)
            self.assertIn(
                json.dumps(repair["residual_improvement"]), serialized
            )

    def test_renderer_writes_same_five_flat_mp4s_each_run(self):
        records = [_record("scene", track_id) for track_id in range(14)]
        state = {
            "trajectory_pattern_records": records,
            "relative_object_motion": [{"video_id": "scene", "frames": []}],
            "pre_pattern_relative_object_motion": [
                {"video_id": "scene", "frames": []}
            ],
        }

        def fake_render(*_args, **kwargs):
            output_path = Path(kwargs["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return str(output_path), "rendered"

        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception.trajectory_pattern_visualization."
            "_render_step8bc_track_video",
            side_effect=fake_render,
        ) as render:
            first_root = Path(tmp) / "first"
            second_root = Path(tmp) / "second"
            log_output = io.StringIO()
            with redirect_stdout(log_output):
                first_manifest = render_step8bc_track_videos(
                    state,
                    first_root,
                    fps=7.5,
                    max_tracks_per_video=10,
                )
                reversed_state = {
                    **state,
                    "trajectory_pattern_records": list(reversed(records)),
                }
                second_manifest = render_step8bc_track_videos(
                    reversed_state,
                    second_root,
                    fps=7.5,
                    max_tracks_per_video=10,
                )
            for manifest in (first_manifest, second_manifest):
                self.assertEqual(
                    manifest["layout"],
                    "scene_left_ego_states_middle_track_signals_right",
                )
                self.assertEqual(
                    manifest["canvas_resolution"], [2480, 1440]
                )
                self.assertEqual(manifest["canvas_aspect_ratio"], "31:18")
                self.assertEqual(manifest["process_panel_width"], 820)
                self.assertEqual(manifest["ego_state_panel_width"], 820)
                self.assertEqual(
                    manifest["middle_panel_role"],
                    "step7f_final_ego_label_state_timeline",
                )
                self.assertEqual(manifest["track_signal_panel_width"], 560)
                self.assertFalse(manifest["scene_bbox_labels"])
                self.assertEqual(
                    manifest["track_progress_position"],
                    "directly_below_scene",
                )
                self.assertEqual(
                    manifest["progress_colors"]["modified_or_added"],
                    "green",
                )
                self.assertIsNone(manifest["max_tracks_per_video"])
                self.assertEqual(manifest["max_visualization_videos_total"], 5)
            for marker in (
                "MP4_START",
                "MP4_TRACK_START",
                "MP4_TRACK_DONE",
                "MP4_DONE",
            ):
                self.assertIn(marker, log_output.getvalue())

            self.assertEqual(render.call_count, 10)
            first_mp4s = sorted(
                path.relative_to(first_root)
                for path in first_root.glob("*_track_*_8b_8c.mp4")
            )
            second_mp4s = sorted(
                path.relative_to(second_root)
                for path in second_root.glob("*_track_*_8b_8c.mp4")
            )
            self.assertEqual(first_mp4s, second_mp4s)
            self.assertEqual(len(first_mp4s), 5)
            self.assertEqual({path.parent for path in first_root.glob("*_track_*_8b_8c.mp4")}, {first_root})
            for root in (first_root, second_root):
                files = [path for path in root.rglob("*") if path.is_file()]
                self.assertEqual(len(files), 5)
                self.assertEqual({path.suffix.lower() for path in files}, {".mp4"})


if __name__ == "__main__":
    unittest.main()
