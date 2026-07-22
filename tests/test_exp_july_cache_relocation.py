import json
import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.pipeline import (
    relocate_cached_payload,
    relocate_json_cache_file,
)


class CopiedCacheRelocationTests(unittest.TestCase):
    def test_remote_dataset_and_pipeline_paths_relocate_to_existing_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "local_dataset"
            pipeline_root = root / "local_output" / "pipeline_output"
            video_id = "video_a"

            frame_path = dataset_root / "frames" / video_id / "frame_00001.jpg"
            depth_path = dataset_root / "depth_maps" / video_id / "frame_00001_depth.npz"
            positions_path = (
                pipeline_root
                / "06_driving_mini_3d_positions"
                / video_id
                / "positions_3d.json"
            )
            for path in (frame_path, depth_path, positions_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            payload = {
                "video_id": video_id,
                "frames": [
                    {
                        "image_path": f"/remote/dataset/driving_mini/frames/{video_id}/{frame_path.name}",
                        "depth_map_path": f"/remote/dataset/driving_mini/depth_maps/{video_id}/{depth_path.name}",
                    }
                ],
                "positions_json": (
                    f"/remote/server/CauVid_output/pipeline_output/"
                    f"06_driving_mini_3d_positions/{video_id}/positions_3d.json"
                ),
            }
            relocated, changes = relocate_cached_payload(
                payload,
                dataset_root=dataset_root,
                pipeline_root=pipeline_root,
            )

            self.assertEqual(relocated["frames"][0]["image_path"], str(frame_path))
            self.assertEqual(relocated["frames"][0]["depth_map_path"], str(depth_path))
            self.assertEqual(relocated["positions_json"], str(positions_path))
            self.assertEqual(len(changes), 3)

    def test_cache_file_is_rewritten_once_and_unknown_missing_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            pipeline_root = root / "pipeline_output"
            video_id = "video_b"
            frame_path = dataset_root / "frames" / video_id / "frame_00002.jpg"
            frame_path.parent.mkdir(parents=True)
            frame_path.touch()

            cache_path = root / "tracks.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "video_id": video_id,
                        "frames": [
                            {
                                "image_path": f"/dataset/driving_mini/frames/{video_id}/{frame_path.name}",
                                "unrelated_path": "/remote/file/that/was/not/copied.bin",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            relocated, changes = relocate_json_cache_file(
                cache_path,
                dataset_root=dataset_root,
                pipeline_root=pipeline_root,
            )
            self.assertEqual(relocated["frames"][0]["image_path"], str(frame_path))
            self.assertEqual(
                relocated["frames"][0]["unrelated_path"],
                "/remote/file/that/was/not/copied.bin",
            )
            self.assertEqual(len(changes), 1)

            _, second_changes = relocate_json_cache_file(
                cache_path,
                dataset_root=dataset_root,
                pipeline_root=pipeline_root,
            )
            self.assertEqual(second_changes, [])


if __name__ == "__main__":
    unittest.main()
