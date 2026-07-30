from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from src.external_depth_anything import depth_map_generator


class _FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def inference(self, paths):
        count = len(paths)
        return SimpleNamespace(
            depth=np.ones((count, 2, 2), dtype=np.float32),
            conf=None,
        )


def test_depth_model_is_loaded_once_per_process_device():
    depth_map_generator.clear_depth_model_cache()
    fake = _FakeModel()
    loader = mock.Mock(return_value=fake)
    fake_api = SimpleNamespace(from_pretrained=loader)
    with mock.patch.object(depth_map_generator, "DepthAnything3", fake_api):
        first, first_hit = depth_map_generator._get_cached_depth_model(
            "test/model", depth_map_generator.torch.device("cpu")
        )
        second, second_hit = depth_map_generator._get_cached_depth_model(
            "test/model", depth_map_generator.torch.device("cpu")
        )

    assert first is second
    assert first_hit is False
    assert second_hit is True
    loader.assert_called_once_with("test/model")


def test_existing_depth_maps_are_filtered_before_model_load(tmp_path):
    input_dir = tmp_path / "frames"
    output_dir = tmp_path / "depth"
    input_dir.mkdir()
    output_dir.mkdir()
    Image.new("RGB", (2, 2)).save(input_dir / "frame_00001.jpg")
    np.savez_compressed(
        output_dir / "frame_00001_depth.npz",
        depth=np.ones((2, 2), dtype=np.float32),
    )

    with mock.patch.object(
        depth_map_generator,
        "_get_cached_depth_model",
    ) as load:
        result = depth_map_generator.generate_depth_maps(
            input_dir,
            output_dir,
            device="cpu",
            quiet=True,
            skip_existing=True,
        )

    assert result == {"processed": 0, "model_cache_hit": None}
    load.assert_not_called()
