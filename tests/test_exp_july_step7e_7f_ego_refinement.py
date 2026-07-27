import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.exp_july.perception.ego_symbol_finalization import build_html, finalize_video, render_mp4s
from src.exp_july.perception.ego_threshold_label_refinement import refine_video
from src.exp_july.perception.pipeline import (
    step7e_threshold_label_refinement,
    step7f_ego_symbol_finalization,
)


def _vector(index):
    x = 20.0 + 12.0 * index
    return {
        "patch_id": index,
        "region_id": f"r{index % 3}c{index % 3}",
        "start_xy": [x, 40.0 + index],
        "end_xy": [x + 2.0, 40.0 + index],
        "dx": 2.0,
        "dy": 0.0,
        "magnitude": 2.0,
        "radial_projection": 1.5,
        "radial_state": "expansion",
        "forward_backward_error": 0.03,
        "local_vector_residual": 0.08,
    }


def _raw():
    vectors = [_vector(i) for i in range(9)]
    pairs = []
    for frame in range(4):
        pairs.append({
            "start_frame": frame,
            "end_frame": frame + 1,
            "status": "completed",
            "raw_patch_count": len(vectors),
            "patch_vectors": vectors,
        })
    return {
        "version": 1,
        "video_id": "demo",
        "configuration": {"region_rows": 3, "region_cols": 3},
        "segments": [{
            "segment_id": 0,
            "provisional_action": "backward",
            "start_frame": 0,
            "end_frame": 4,
            "frame_pair_evidence": pairs,
            "patch_vectors": vectors * 4,
        }],
    }


def _candidate(candidate_id, action, threshold):
    return {
        "candidate_id": candidate_id,
        "thresholds": {
            "static_speed_threshold": threshold,
            "lateral_threshold": 0.2,
            "yaw_threshold": 0.03,
        },
        "actions": [action] * 5,
        "segments": [{
            "segment_id": 0,
            "action": action,
            "start_frame": 0,
            "end_frame": 4,
            "duration_frames": 5,
        }],
        "score_components": {"action_complexity": 0.1, "signal_fit_error": 0.1},
        "num_rapid_left_right_reversals": 0,
        "num_short_segments": 0,
    }


def _provisional():
    signals = [
        {
            "frame_index": frame,
            "available": True,
            "ego_vx": 0.0,
            "ego_vz": -1.0,
            "ego_yaw_rate": 0.0,
            "ego_speed": 1.0,
            "ego_speed_delta": 0.0,
        }
        for frame in range(5)
    ]
    return {
        "video_id": "demo",
        "label_status": "provisional",
        "selected_thresholds": {"static_speed_threshold": 0.2, "lateral_threshold": 0.2, "yaw_threshold": 0.03},
        "configuration": {"acceleration_threshold": 0.12},
        "continuous_signals": signals,
        "frames": [{"frame_index": frame, "action": "backward"} for frame in range(5)],
        "final_action_segments": [{"segment_id": 0, "action": "backward", "start_frame": 0, "end_frame": 4, "duration_frames": 5}],
    }


class Step7E7FEgoRefinementTests(unittest.TestCase):
    def test_backward_is_corrected_to_forward_by_rule_rank_and_stabilizes(self):
        result = refine_video(
            "demo",
            [_candidate("backward", "backward", 0.2), _candidate("forward", "forward", 0.3)],
            _raw(),
            _provisional(),
        )
        self.assertTrue(result["stabilized"])
        self.assertEqual(result["stop_reason"], "thresholds_and_labels_stable")
        self.assertEqual(result["selected_candidate_id"], "forward")
        self.assertEqual(result["refined_segments"][0]["action"], "forward")
        self.assertEqual(result["refined_segments"][0]["validation_status"], "validated")
        self.assertEqual(result["corrections"][0]["provisional_actions"], ["backward"])
        self.assertEqual(result["corrections"][0]["corrected_action"], "forward")
        self.assertLess(
            result["candidate_rankings"][0]["hard_rule_violations"],
            result["candidate_rankings"][1]["hard_rule_violations"],
        )

    def test_finalization_publishes_only_validated_labels_and_preserves_audit(self):
        refinement = refine_video(
            "demo", [_candidate("forward", "forward", 0.3)], _raw(), _provisional()
        )
        final = finalize_video(refinement, _provisional())
        self.assertEqual(final["label_status"], "final")
        self.assertTrue(final["downstream_usable_as_final"])
        self.assertEqual(final["frames"][0]["action"], "forward")
        self.assertGreater(final["frames"][0]["observable_cues"]["ego_driving_forward"], 0)
        self.assertEqual(final["provisional_segments"][0]["action"], "backward")
        self.assertTrue(final["final_action_segments"][0]["fired_rule_ids"])
        self.assertIn("Corrected provisional backward to forward", final["final_action_segments"][0]["correction_reason"])
        self.assertIn("candidate_rankings", final)
        self.assertIn("normalized_evidence", final)

    def test_uncertain_segment_is_not_materialized_as_normal_motion(self):
        raw = {"version": 1, "video_id": "demo", "configuration": {}, "segments": []}
        refinement = refine_video("demo", [_candidate("forward", "forward", 0.3)], raw, _provisional())
        self.assertEqual(refinement["refined_segments"][0]["validation_status"], "uncertain")
        final = finalize_video(refinement, _provisional())
        self.assertEqual(final["frames"][0]["action"], "unknown")
        self.assertEqual(final["frames"][0]["observable_cues"]["ego_motion_uncertain"], 1.0)
        self.assertEqual(final["frames"][0]["observable_cues"]["ego_driving_forward"], 0.0)

    def test_pipeline_stages_cache_and_replace_provisional_downstream_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = _provisional()
            prior["configuration"].update({
                "candidate_static_speed_thresholds": [0.2],
                "candidate_lateral_thresholds": [0.2],
                "candidate_yaw_thresholds": [0.03],
                "max_candidates": 1,
            })
            rule_state = {
                "ego_symbol_prior": [prior],
                "background_motion_evidence": [_raw()],
            }
            frames = []
            for index in range(5):
                image = np.full((120, 180, 3), (25, 35, 45), dtype=np.uint8)
                image_path = root / f"pipeline_frame_{index:05d}.png"
                cv2.imwrite(str(image_path), image)
                frames.append({"frame_index": index, "image_path": str(image_path)})
            position_state = {"positions_3d": [{"video_id": "demo", "frames": frames}]}
            output_root = root / "outputs"
            with patch(
                "src.exp_july.perception.pipeline.get_pipeline_output_root",
                return_value=output_root,
            ):
                refined_first = step7e_threshold_label_refinement(rule_state)
                refined_cached = step7e_threshold_label_refinement(rule_state)
                final_first = step7f_ego_symbol_finalization(position_state, refined_cached)
                final_cached = step7f_ego_symbol_finalization(position_state, refined_cached)
            self.assertEqual(refined_first["ego_threshold_label_refinement_manifest"]["cached_videos"], 0)
            self.assertEqual(refined_cached["ego_threshold_label_refinement_manifest"]["cached_videos"], 1)
            self.assertEqual(final_first["final_ego_symbol_manifest"]["cached_videos"], 0)
            self.assertEqual(final_cached["final_ego_symbol_manifest"]["cached_videos"], 1)
            self.assertEqual(final_cached["ego_symbol_prior"][0]["label_status"], "final")
            self.assertEqual(final_cached["provisional_ego_symbol_prior"][0]["label_status"], "provisional")
            self.assertTrue(Path(final_cached["ego_symbol_audit_html_path"]).exists())
            self.assertTrue(final_cached["ego_symbol_audit_mp4_manifest"]["rendered"][0]["cache_hit"])

    def test_offline_html_and_mp4_audit_are_generated(self):
        refinement = refine_video("demo", [_candidate("forward", "forward", 0.3)], _raw(), _provisional())
        final = finalize_video(refinement, _provisional())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for index in range(5):
                image = np.full((120, 180, 3), (25, 35, 45), dtype=np.uint8)
                path = root / f"frame_{index:05d}.png"
                cv2.imwrite(str(path), image)
                frames.append({"frame_index": index, "image_path": str(path)})
            html_path = build_html([final], root / "audit")
            videos = render_mp4s([final], {"demo": {"video_id": "demo", "frames": frames}}, root / "videos", fps=5)
            self.assertTrue(Path(html_path).exists())
            text = Path(html_path).read_text(encoding="utf-8")
            self.assertIn("Final Ego Symbol Audit", text)
            self.assertIn("threshold_changes", text)
            self.assertEqual(len(videos["rendered"]), 1)
            self.assertTrue(Path(videos["rendered"][0]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
