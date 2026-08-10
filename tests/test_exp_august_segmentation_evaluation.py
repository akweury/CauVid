import json
import tempfile
import unittest
from pathlib import Path

from src.exp_august.evaluation import Segment
from src.exp_august.evaluation import VideoTimeline
from src.exp_august.evaluation import deterministic_split
from src.exp_august.evaluation import evaluate_dataset
from src.exp_august.evaluation import evaluate_video
from src.exp_august.evaluation import load_annotation
from src.exp_august.evaluation_report import write_test_evaluation_pdf


def timeline(video_id, num_frames, segments):
    return VideoTimeline(
        video_id=video_id,
        num_frames=num_frames,
        segments=tuple(Segment(*row) for row in segments),
    )


class ExpAugustSegmentationEvaluationTests(unittest.TestCase):
    def test_perfect_prediction(self):
        gt = timeline("video", 20, [(0, 4, "forward"), (5, 9, "left"), (10, 19, "static")])
        result = evaluate_video(gt, gt)

        frame = result["frame_classification"]
        self.assertEqual(frame["accuracy"], 1.0)
        self.assertEqual(frame["weighted_f1"], 1.0)
        self.assertEqual(result["boundary_detection"]["1"]["f1"], 1.0)
        self.assertEqual(result["segment_evaluation"]["mean_matched_iou"], 1.0)
        self.assertEqual(result["segment_evaluation"]["label_aware_segment_iou"], 1.0)

    def test_completely_wrong_prediction(self):
        gt = timeline("video", 10, [(0, 4, "forward"), (5, 9, "left")])
        prediction = timeline("video", 10, [(0, 4, "backward"), (5, 9, "right")])
        result = evaluate_video(gt, prediction)

        self.assertEqual(result["frame_classification"]["accuracy"], 0.0)
        self.assertEqual(result["segment_evaluation"]["mean_matched_iou"], 1.0)
        self.assertEqual(result["segment_evaluation"]["label_aware_segment_iou"], 0.0)

    def test_shifted_boundaries_respect_tolerance_and_one_to_one_matching(self):
        gt = timeline("video", 20, [(0, 4, "forward"), (5, 9, "left"), (10, 19, "static")])
        prediction = timeline("video", 20, [(0, 6, "forward"), (7, 11, "left"), (12, 19, "static")])
        result = evaluate_video(gt, prediction, tolerances=(1, 3, 5, 10))

        self.assertEqual(result["boundary_detection"]["1"]["f1"], 0.0)
        self.assertEqual(result["boundary_detection"]["3"]["f1"], 1.0)
        self.assertEqual(result["boundary_detection"]["3"]["tp"], 2)

    def test_missing_segments_are_penalized(self):
        gt = timeline("video", 30, [(0, 9, "forward"), (10, 19, "left"), (20, 29, "static")])
        prediction = timeline("video", 30, [(0, 29, "forward")])
        result = evaluate_video(gt, prediction)

        self.assertLess(result["frame_classification"]["accuracy"], 1.0)
        self.assertEqual(result["segment_evaluation"]["num_predicted_segments"], 1)
        self.assertLess(result["segment_evaluation"]["segment_iou"], 1.0)
        self.assertEqual(result["boundary_detection"]["10"]["recall"], 0.0)

    def test_unequal_video_lengths_are_aligned_by_full_duration(self):
        gt = timeline("video", 12, [(0, 5, "forward"), (6, 11, "static")])
        prediction = timeline("video", 6, [(0, 2, "forward"), (3, 5, "static")])
        result = evaluate_video(gt, prediction)

        self.assertEqual(result["alignment"]["method"], "normalized_full_duration")
        self.assertEqual(result["alignment"]["scale"], 2.0)
        self.assertEqual(result["frame_classification"]["accuracy"], 1.0)
        self.assertEqual(result["segment_evaluation"]["label_aware_segment_iou"], 1.0)

    def test_repository_annotation_schema_is_detected(self):
        path = Path("annotations/video_segmentation/videos__0001542f-ec815219-c43048a090.json")
        annotation, error = load_annotation(path)
        self.assertIsNone(error)
        self.assertEqual(annotation.video_id, "0001542f-ec815219")
        self.assertEqual(annotation.num_frames, 1208)
        self.assertEqual([row.label for row in annotation.segments], ["right", "forward", "static"])

    def test_split_is_deterministic_disjoint_and_seeded(self):
        video_ids = [f"video-{index}" for index in range(20)]
        first = deterministic_split(video_ids, seed=7, test_ratio=0.25)
        second = deterministic_split(list(reversed(video_ids)), seed=7, test_ratio=0.25)
        different = deterministic_split(video_ids, seed=8, test_ratio=0.25)

        self.assertEqual(first, second)
        self.assertEqual(len(first["test"]), 5)
        self.assertFalse(set(first["test"]) & set(first["dev"]))
        self.assertNotEqual(first["test"], different["test"])

    def test_dataset_matching_reports_matched_missing_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations"
            predictions = root / "predictions"
            output = root / "evaluation"
            annotations.mkdir()
            predictions.mkdir()

            def annotation(video_id, valid=True):
                payload = {
                    "schema_version": 1,
                    "video": {"id": f"videos__{video_id}-hash", "path": f"videos/{video_id}.mov", "fps": 30.0, "frame_count": 10},
                    "segments": [{"start_frame": 0, "end_frame": 9, "label": "moving_forward"}],
                    "keyframes": [{"frame": 0, "label": "moving_forward"}],
                }
                if not valid:
                    payload["segments"][0]["label"] = "not_a_motion_label"
                (annotations / f"{video_id}.json").write_text(json.dumps(payload), encoding="utf-8")

            annotation("video-a")
            annotation("video-b")
            annotation("video-invalid", valid=False)
            prediction = {
                "video_id": "video-a",
                "num_frames": 5,
                "segments": [{"start_frame": 0, "end_frame": 4, "event": "forward_static_moving|straightforward"}],
            }
            (predictions / "temporal_segmentation.json").write_text(json.dumps(prediction), encoding="utf-8")

            results = evaluate_dataset(predictions, annotations, output, split="all", seed=3, test_ratio=0.5)
            report = write_test_evaluation_pdf(results, output / "step_08_test_evaluation_charts.pdf")

            self.assertEqual(results["matching"]["valid_annotations"], 2)
            self.assertEqual(results["matching"]["invalid_annotations"], 1)
            self.assertEqual(results["matching"]["matched"], 1)
            self.assertEqual(results["matching"]["missing_predictions"], 1)
            for filename in (
                "evaluation_results.json",
                "per_video_metrics.csv",
                "per_video_class_metrics.csv",
                "aggregate_metrics.csv",
                "boundary_metrics.csv",
                "segment_matches.csv",
                "confusion_matrix.csv",
                "confusion_matrix.svg",
                "metric_summary.svg",
                "split_manifest.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertTrue(report.is_file())
            self.assertGreater(report.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
