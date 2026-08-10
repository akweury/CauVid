import tempfile
import unittest
from pathlib import Path

from src.exp_august.splits import create_split_manifest, load_or_create_split_manifest


class ExpAugustSplitTests(unittest.TestCase):
    def test_standard_scale_counts_are_70_15_15(self):
        cases = {
            10: {"train": 7, "eval": 2, "test": 1},
            100: {"train": 70, "eval": 15, "test": 15},
            961: {"train": 673, "eval": 144, "test": 144},
        }
        videos = [f"video-{index:04d}" for index in range(961)]
        for scale, expected in cases.items():
            manifest = create_split_manifest(videos, scale, seed=726381)
            self.assertEqual(manifest["counts"], expected)
            assignments = [set(manifest[f"{name}_video_ids"]) for name in ("train", "eval", "test")]
            self.assertFalse(assignments[0] & assignments[1])
            self.assertFalse(assignments[0] & assignments[2])
            self.assertFalse(assignments[1] & assignments[2])

    def test_existing_manifest_is_authoritative_for_same_seed_and_scale(self):
        videos = [f"video-{index:03d}" for index in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_split_manifest.json"
            first = load_or_create_split_manifest(path, videos, 10, 184957)
            second = load_or_create_split_manifest(path, list(reversed(videos)) + ["new-video"], 10, 184957)
        self.assertEqual(first, second)

    def test_existing_manifest_rejects_seed_or_scale_mismatch(self):
        videos = [f"video-{index:03d}" for index in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_split_manifest.json"
            load_or_create_split_manifest(path, videos, 10, 726381)
            with self.assertRaises(ValueError):
                load_or_create_split_manifest(path, videos, 11, 726381)
            with self.assertRaises(ValueError):
                load_or_create_split_manifest(path, videos, 10, 184957)

    def test_scale_cannot_silently_use_fewer_videos_than_requested(self):
        with self.assertRaisesRegex(ValueError, "Requested 10 videos"):
            create_split_manifest(["a", "b"], 10, 726381)

    def test_test_partition_contains_only_annotated_videos(self):
        videos = [f"video-{index:04d}" for index in range(100)]
        annotated = set(videos[::2])
        manifest = create_split_manifest(videos, 100, 726381, annotated)
        self.assertEqual(manifest["counts"], {"train": 70, "eval": 15, "test": 15})
        self.assertTrue(set(manifest["test_video_ids"]) <= annotated)

    def test_full_scale_caps_test_at_available_annotations(self):
        videos = [f"video-{index:04d}" for index in range(961)]
        annotated = set(videos[:101])
        manifest = create_split_manifest(videos, 961, 726381, annotated)
        self.assertEqual(manifest["counts"], {"train": 716, "eval": 144, "test": 101})
        self.assertEqual(set(manifest["test_video_ids"]), annotated)
        self.assertEqual(manifest["requested_counts"], {"train": 673, "eval": 144, "test": 144})


if __name__ == "__main__":
    unittest.main()
