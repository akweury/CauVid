from pathlib import Path

import cv2
import numpy as np

from src.exp_july.perception.important_objects_visualization import render_step11_video


def test_step11_renders_scene_ego_timeline_and_four_object_slots(tmp_path: Path):
    frames = []
    for frame_index in range(3):
        image_path = tmp_path / f"frame_{frame_index:05d}.jpg"
        cv2.imwrite(str(image_path), np.full((240, 420, 3), 35 + frame_index * 15, np.uint8))
        objects = []
        for track_id in range(5):
            objects.append(
                {
                    "track_id": track_id,
                    "label": "car",
                    "bbox": [20 + track_id * 25, 40, 90 + track_id * 25, 150],
                    "is_observed": True,
                    "has_rel_motion": frame_index > 0,
                    "score": 0.9 - track_id * 0.05,
                    "vx_state": "relative_right",
                    "vz_state": "approaching",
                    "speed_state": "moving",
                    "x_position_state": "center",
                    "distance_state": "near",
                    "rel_vx": 0.1,
                    "rel_vz": -0.2,
                    "rel_speed": 0.224,
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "image_path": str(image_path),
                "objects": objects,
            }
        )
    relative_video = {"video_id": "demo", "frames": frames}
    ego_result = {
        "video_id": "demo",
        "final_segmentation": {
            "vx": {
                "frames": [
                    {"frame_index": index, "state": "straight", "confidence": 0.9}
                    for index in range(3)
                ]
            },
            "vz": {
                "frames": [
                    {"frame_index": index, "state": "forward", "confidence": 0.9}
                    for index in range(3)
                ]
            },
        },
    }
    output_path = tmp_path / "step11.mp4"

    result = render_step11_video(relative_video, ego_result, output_path, fps=5.0)

    assert result["status"] == "rendered"
    assert result["max_objects_per_frame"] == 4
    assert output_path.is_file()
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1080
    finally:
        capture.release()
