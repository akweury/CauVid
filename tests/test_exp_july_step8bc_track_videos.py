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
    build_step8bc_track_video_payload,
    render_step8bc_track_videos,
    render_trajectory_pattern_visualizations,
    select_deterministic_track_records,
)


def _record(video_id, track_id):
    step_metrics = {
        "max_bbox_center_step_px_per_frame": 10.0 + track_id,
        "max_bbox_center_step_diag_ratio": 0.10 + track_id / 1000.0,
        "max_bbox_size_ratio": 1.10 + track_id / 1000.0,
        "max_depth_step_per_frame": 0.20 + track_id / 1000.0,
        "raw_max_depth_step_per_frame": 0.30 + track_id / 1000.0,
        "max_rel_velocity_delta": 0.40 + track_id / 1000.0,
        "max_rel_speed": 0.50 + track_id / 1000.0,
        "legacy_max_rel_velocity_delta": 0.60 + track_id / 1000.0,
        "legacy_max_rel_speed": 0.70 + track_id / 1000.0,
        "direction_reversal_count": track_id % 3,
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
            "source_decision": "Keep",
            "source_validation": {
                "validation_status": "valid",
                "step_metrics": step_metrics,
                "thresholds": {
                    "max_invalid_rel_speed": 20.0,
                    "max_uncertain_rel_speed": 10.0,
                },
                "issues": [
                    {
                        "kind": "speed_abnormal_change",
                        "severity": "uncertain",
                        "value": 9.875,
                        "threshold": 10.0,
                    }
                ],
                "checks": {
                    "speed_abnormal_change": {
                        "passed": True,
                        "max_validation_speed": step_metrics["max_rel_speed"],
                    }
                },
                "ego_motion_consistency": {
                    "selected_velocity_profile": "ego_minus",
                    "velocity_profiles": [
                        {"name": "ego_minus", "max_speed": 2.125},
                        {"name": "ego_plus", "max_speed": 3.375},
                    ],
                },
            },
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
    }


class Step8BCTrackVideoTests(unittest.TestCase):
    def test_pattern_reports_are_offline_html_complete_and_escaped(self):
        record = _record("scene", 7)
        record["symbolic_track"]["object_class"] = "<script>bad()</script>"
        record["llm_residual_interpretation"] = [
            {
                "pattern_id": pattern_id,
                "plausibility": 0.5,
                "structural_conflicts": [],
                "explanation": (
                    "<script>explanation()</script>"
                    if pattern_id == "approaching"
                    else f"full explanation for {pattern_id}"
                ),
            }
            for pattern_id in PATTERNS
        ]
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
            result = render_trajectory_pattern_visualizations(state, root)
            track_path = (
                root / "scene" / "track_0007_pattern_process.html"
            )
            summary_path = root / "scene" / "video_pattern_summary.html"
            self.assertTrue(track_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertFalse(list(root.rglob("*.png")))

            track_html = track_path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", track_html)
            self.assertNotIn("<script>bad()</script>", track_html)
            self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", track_html)
            self.assertNotIn("<script>explanation()</script>", track_html)
            self.assertIn(
                "&lt;script&gt;explanation()&lt;/script&gt;", track_html
            )
            self.assertNotIn("http://", track_html)
            self.assertNotIn("https://", track_html)
            for section_id in (
                "symbolic-track",
                "pattern-residuals",
                "llm-interpretation",
                "repair-candidates",
                "symbolic-validation",
                "final-result",
                "provenance",
            ):
                self.assertIn(f'id="{section_id}"', track_html)
            for pattern_id in PATTERNS:
                self.assertIn(f'data-pattern="{pattern_id}"', track_html)
            for residual_id in RESIDUALS:
                self.assertIn(f'data-residual="{residual_id}"', track_html)
            for repair in record["candidate_repairs"]:
                self.assertIn(
                    f'data-candidate-id="{repair["candidate_id"]}"',
                    track_html,
                )

            summary_html = summary_path.read_text(encoding="utf-8")
            self.assertIn("track_0007_pattern_process.html", summary_html)
            for pattern_id in PATTERNS:
                self.assertIn(f'data-pattern="{pattern_id}"', summary_html)

            manifest = json.loads(
                (
                    root / "trajectory_pattern_visualization_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["version"], 4)
            self.assertEqual(manifest["report_format"], "html")
            self.assertEqual(manifest["num_track_reports"], 1)
            self.assertEqual(manifest["num_summary_reports"], 1)
            self.assertEqual(
                result["trajectory_pattern_visualizations"][0][
                    "media_type"
                ],
                "text/html",
            )

    def test_selection_is_stable_order_independent_and_capped_per_video(self):
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
        counts = Counter(video_id for video_id, _track_id in selected_keys)
        self.assertEqual(counts, {"scene_a": 6, "scene_b": 10})

    def test_payload_keeps_every_8b_metric_and_every_8c_residual_distance(self):
        record = _record("scene", 7)

        payload = build_step8bc_track_video_payload(record)

        self.assertEqual(payload["video_id"], "scene")
        self.assertEqual(payload["track_id"], 7)
        serialized = json.dumps(payload, sort_keys=True)
        step_metrics = record["symbolic_track"]["source_validation"]["step_metrics"]
        for metric_name, metric_value in step_metrics.items():
            self.assertIn(metric_name, serialized)
            self.assertIn(json.dumps(metric_value), serialized)
        for required_8b_field in (
            "max_frame_gap",
            "has_motion_ratio",
            "checks",
            "issues",
            "thresholds",
            "ego_minus",
            "ego_plus",
        ):
            self.assertIn(required_8b_field, serialized)
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

    def test_renderer_writes_same_ten_track_folders_mp4s_and_metrics_each_run(self):
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
                render_step8bc_track_videos(
                    state,
                    first_root,
                    fps=7.5,
                    max_tracks_per_video=10,
                )
                reversed_state = {
                    **state,
                    "trajectory_pattern_records": list(reversed(records)),
                }
                render_step8bc_track_videos(
                    reversed_state,
                    second_root,
                    fps=7.5,
                    max_tracks_per_video=10,
                )
            for marker in (
                "MP4_START",
                "MP4_TRACK_START",
                "MP4_TRACK_DONE",
                "MP4_DONE",
            ):
                self.assertIn(marker, log_output.getvalue())

            self.assertEqual(render.call_count, 20)
            first_mp4s = sorted(
                path.relative_to(first_root)
                for path in first_root.glob("scene/track_*/*.mp4")
            )
            second_mp4s = sorted(
                path.relative_to(second_root)
                for path in second_root.glob("scene/track_*/*.mp4")
            )
            self.assertEqual(first_mp4s, second_mp4s)
            self.assertEqual(len(first_mp4s), 10)
            for relative_path in first_mp4s:
                track_folder = relative_path.parent
                track_id = int(track_folder.name.removeprefix("track_"))
                self.assertEqual(
                    relative_path.name, f"track_{track_id:04d}_8b_8c.mp4"
                )
                metrics_files = list(
                    (first_root / track_folder).glob("*.json")
                )
                self.assertEqual(len(metrics_files), 1)
                metrics = json.loads(
                    metrics_files[0].read_text(encoding="utf-8")
                )
                self.assertEqual(metrics["video_id"], "scene")
                self.assertEqual(metrics["track_id"], track_id)
                serialized = json.dumps(metrics, sort_keys=True)
                for residual_name in RESIDUALS:
                    self.assertIn(residual_name, serialized)


if __name__ == "__main__":
    unittest.main()
