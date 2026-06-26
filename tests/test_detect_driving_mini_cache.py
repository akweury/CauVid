from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.exp_driving_videos.modules import detect_driving_mini


class DetectDrivingMiniCacheTest(unittest.TestCase):
    @staticmethod
    def _cached_frame() -> dict:
        return {
            "frame": "frame_00001.jpg",
            "frame_index": 1,
            "image_path": "/tmp/frame_00001.jpg",
            "accepted_detections": [
                {
                    "detection_id": "000001:accepted:0000",
                    "class": "car",
                    "score": 0.95,
                    "raw_score": 0.95,
                    "calibrated_score": 0.95,
                    "feedback_bonus": 0.0,
                    "score_used_for_candidate_ranking": 0.95,
                    "accepted": True,
                }
            ],
            "candidate_detections": [
                {
                    "detection_id": "000001:candidate:0000",
                    "class": "pedestrian",
                    "score": 0.22,
                    "raw_score": 0.22,
                    "calibrated_score": 0.22,
                    "feedback_bonus": 0.0,
                    "score_used_for_candidate_ranking": 0.22,
                    "accepted": False,
                    "candidate_source": "low_confidence",
                }
            ],
            "od_calibration": {
                "policy_id": "",
                "policy_version": 0,
                "policy_available": False,
            },
        }

    def test_process_video_cache_hit_is_read_only_when_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            out_dir = output_root / "video_cache"
            out_dir.mkdir(parents=True, exist_ok=True)
            detections_file = out_dir / "detections.json"
            predictions_csv_file = out_dir / "detection_predictions.csv"
            class_summary_csv_file = out_dir / "detection_class_summary.csv"
            predictions_csv_file.write_text("header\n", encoding="utf-8")
            class_summary_csv_file.write_text("header\n", encoding="utf-8")
            payload = {
                "schema_version": 6,
                "video_id": "video_cache",
                "frames": [self._cached_frame()],
                "od_calibration": {
                    "policy_id": "",
                    "policy_version": 0,
                    "policy_available": False,
                },
                "output_paths": {},
            }
            original_text = json.dumps(payload, indent=2)
            detections_file.write_text(original_text, encoding="utf-8")

            with mock.patch.object(detect_driving_mini, "_ensure_detection_identity_and_calibration") as ensure_mock, \
                mock.patch.object(detect_driving_mini, "_build_prediction_rows") as predictions_mock, \
                mock.patch.object(detect_driving_mini, "_build_class_summary_rows") as summary_mock, \
                mock.patch.object(detect_driving_mini, "_write_csv") as write_csv_mock, \
                mock.patch.object(detect_driving_mini, "render_detection_video") as render_mock:
                result = detect_driving_mini.process_video(
                    video_id="video_cache",
                    detector=None,
                    output_root=output_root,
                    force_recompute=False,
                    render_video=False,
                )

            self.assertTrue(result["from_cache"])
            self.assertFalse(result.get("_requires_detection", False))
            ensure_mock.assert_not_called()
            predictions_mock.assert_not_called()
            summary_mock.assert_not_called()
            write_csv_mock.assert_not_called()
            render_mock.assert_not_called()
            self.assertEqual(detections_file.read_text(encoding="utf-8"), original_text)

    def test_run_skips_detector_init_when_every_video_is_cached(self) -> None:
        cached_result = {
            "video_id": "video_cache",
            "num_frames": 1,
            "num_detections": 1,
            "num_candidate_detections": 0,
            "num_total_saved_detections": 1,
            "detected_classes": {"car": 1},
            "frames": [self._cached_frame()],
            "from_cache": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            output_root.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(detect_driving_mini, "list_video_ids", return_value=["video_cache"]), \
                mock.patch.object(detect_driving_mini, "process_video", return_value=cached_result), \
                mock.patch.object(detect_driving_mini, "ensure_detector_runtime_available") as ensure_runtime_mock, \
                mock.patch.object(detect_driving_mini, "YOLOWorldDetector") as detector_cls_mock:
                results = detect_driving_mini.run(
                    output_root=output_root,
                    render_video=False,
                )

            self.assertEqual(len(results), 1)
            ensure_runtime_mock.assert_not_called()
            detector_cls_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
