import sys
import unittest
from unittest import mock

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = mock.MagicMock()

from src.exp_nuScenes.detection_pipeline import YOLOWorldDetector


class YOLOWorldDetectorPredictTests(unittest.TestCase):
    def test_predict_batch_uses_ultralytics_half_argument(self) -> None:
        detector = YOLOWorldDetector(device="cuda:0")
        detector._model = mock.Mock()
        detector._model.predict.return_value = []
        detector._runtime_device = "cuda:0"
        detector._resolved_predict_batch_size = 8
        detector._resolved_half_precision = True

        detector._predict_batch(
            ["frame.jpg"],
            conf_threshold=0.3,
            nms_iou_threshold=0.5,
        )

        detector._model.predict.assert_called_once_with(
            source=["frame.jpg"],
            conf=0.3,
            iou=0.5,
            imgsz=640,
            verbose=False,
            batch=8,
            half=True,
            device="cuda:0",
        )
        self.assertNotIn("quantize", detector._model.predict.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
