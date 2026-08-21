import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.exp_august.inference.frames import CanonicalFrameProvider, FrameDecodeError


class CanonicalFrameProviderTests(unittest.TestCase):
    @staticmethod
    def _provider(source_frame_index: int = 2) -> CanonicalFrameProvider:
        provider = object.__new__(CanonicalFrameProvider)
        provider.source_path = Path("/videos/example.mov")
        provider.manifest = SimpleNamespace(
            video_id="example",
            image_size=SimpleNamespace(width=3, height=2),
            frames=(
                SimpleNamespace(
                    frame_index=0,
                    timestamp_s=0.0,
                    source_frame_index=source_frame_index,
                    source_timestamp_s=source_frame_index / 30.0,
                ),
            ),
        )
        return provider

    def test_get_frame_falls_back_to_sequential_decode_when_random_seek_fails(self):
        provider = self._provider()
        random_access = MagicMock()
        random_access.set.return_value = True
        random_access.read.return_value = (False, None)
        sequential = MagicMock()
        sequential.read.side_effect = [
            (True, np.full((2, 3, 3), value, dtype=np.uint8))
            for value in range(3)
        ]

        with patch.object(provider, "_open", side_effect=(random_access, sequential)):
            frame = provider.get_frame(0)

        random_access.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 2.0)
        random_access.release.assert_called_once_with()
        self.assertEqual(sequential.read.call_count, 3)
        sequential.release.assert_called_once_with()
        self.assertTrue(np.all(frame.image_bgr == 2))
        self.assertEqual(frame.source_frame_index, 2)

    def test_get_frame_reports_sequential_failure_after_random_seek_failure(self):
        provider = self._provider()
        random_access = MagicMock()
        random_access.set.return_value = False
        sequential = MagicMock()
        sequential.read.side_effect = [
            (True, np.zeros((2, 3, 3), dtype=np.uint8)),
            (False, None),
        ]

        with patch.object(provider, "_open", side_effect=(random_access, sequential)):
            with self.assertRaisesRegex(
                FrameDecodeError,
                "sequential decode failed at source frame 1 while requesting source frame 2",
            ):
                provider.get_frame(0)

        random_access.read.assert_not_called()
        random_access.release.assert_called_once_with()
        sequential.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
