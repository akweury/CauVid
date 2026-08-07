import json

import pytest

from src.video_annotation.model import AnnotationDocument, Keyframe, VideoMetadata, keyframes_to_segments
from src.video_annotation.storage import (
    load_document,
    load_session,
    save_document,
    save_session,
    session_path,
)
from src.video_annotation.validation import validate_annotation_set, validate_payload
from src.video_annotation.cli import DEFAULT_CONFIG, DEFAULT_SUBSET, main
from src.video_annotation.videos import discover_videos, resolve_video_source
from src.video_annotation.config import load_config
from src.video_annotation.subset import select_manifest_videos


def metadata() -> VideoMetadata:
    return VideoMetadata("sample-abc", "clips/sample.mp4", 10.0, 100)


def test_keyframes_convert_to_inclusive_segments() -> None:
    segments = keyframes_to_segments(
        [Keyframe(50, "turn"), Keyframe(0, "still"), Keyframe(20, "forward")], 100
    )
    assert [(item.start_frame, item.end_frame, item.label) for item in segments] == [
        (0, 19, "still"),
        (20, 49, "forward"),
        (50, 99, "turn"),
    ]


def test_keyframe_conversion_rejects_duplicates_and_bad_ranges() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        keyframes_to_segments([Keyframe(2, "a"), Keyframe(2, "b")], 10)
    with pytest.raises(ValueError, match="outside"):
        keyframes_to_segments([Keyframe(10, "a")], 10)


def test_edit_change_delete_and_navigation() -> None:
    document = AnnotationDocument(metadata(), "tester", "2.0")
    document.set_keyframe(20, "forward", {"forward", "turn"})
    document.set_keyframe(5, "forward", {"forward", "turn"})
    document.set_keyframe(20, "turn", {"forward", "turn"})
    assert document.keyframes == [Keyframe(5, "forward"), Keyframe(20, "turn")]
    assert document.label_at(19) == "forward"
    assert document.label_at(20) == "turn"
    assert document.previous_keyframe(20) == 5
    assert document.next_keyframe(5) == 20
    assert document.delete_keyframe(20)
    assert not document.delete_keyframe(20)


def test_atomic_persistence_exports_keyframes_and_segments(tmp_path) -> None:
    document = AnnotationDocument(metadata(), "alice", "1.3")
    document.set_keyframe(0, "still")
    document.set_keyframe(30, "forward")
    path = save_document(document, tmp_path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["video"]["path"] == "clips/sample.mp4"
    assert raw["keyframes"][1]["timestamp"] == 3.0
    assert raw["segments"][0] == {
        "start_frame": 0,
        "end_frame": 29,
        "start_timestamp": 0.0,
        "end_timestamp": 2.9,
        "label": "still",
    }
    restored = load_document(path)
    assert restored.keyframes == document.keyframes
    assert restored.annotator == "alice"


def test_validation_detects_gap_duplicates_ranges_and_unknown_labels() -> None:
    raw = {
        "video": {"id": "bad", "path": "bad.mp4", "fps": 10, "frame_count": 20},
        "keyframes": [
            {"frame": 3, "label": "known"},
            {"frame": 3, "label": "other"},
            {"frame": 22, "label": "known"},
        ],
    }
    codes = {issue.code for issue in validate_payload(raw, {"known"})}
    assert {"gap", "duplicate_keyframe", "invalid_range", "unknown_label", "invalid_segments"} <= codes


def test_validation_detects_unlabeled_discovered_video(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    video = dataset / "clip.mp4"
    video.touch()
    issues = validate_annotation_set([video], dataset, tmp_path / "annotations", {"still"})
    assert len(issues) == 1
    assert issues[0].code == "unlabeled"


def test_cli_fails_when_dataset_contains_no_supported_videos(tmp_path, capsys) -> None:
    result = main(["validate", "--dataset", str(tmp_path)])
    assert result == 1
    assert "No supported videos" in capsys.readouterr().out


def test_discovery_maps_driving_mini_docker_symlinks(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "driving_mini"
    videos = dataset / "videos"
    raw_root = tmp_path / "raw"
    videos.mkdir(parents=True)
    raw_root.mkdir()
    real_video = raw_root / "clip.mov"
    real_video.write_bytes(b"video-placeholder")
    logical_video = videos / "clip.mov"
    logical_video.symlink_to("/raw_driving_data/clip.mov")
    monkeypatch.setenv("CAUVID_RAW_DRIVING_DATASET", str(raw_root))

    assert discover_videos(dataset, (".mov",)) == [logical_video]
    assert resolve_video_source(logical_video) == real_video


def test_default_config_enables_pipeline_rotation_metadata() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.apply_rotation_metadata is True
    assert config.portrait_fallback_rotation == -90


def test_gui_read_frame_applies_detected_rotation(monkeypatch) -> None:
    import numpy as np
    from src.video_annotation import gui

    original = np.zeros((2, 3, 3), dtype=np.uint8)

    class Capture:
        def read(self):
            return True, original

        def set(self, *_args):
            return True

    annotator = gui.VideoAnnotator.__new__(gui.VideoAnnotator)
    annotator.capture = Capture()
    annotator.document = AnnotationDocument(metadata(), "tester", "1.0")
    annotator.display_rotation = -90
    annotator._next_capture_frame = 0
    annotator.playing = False
    annotator.status = ""
    annotator._set_trackbar = lambda _frame: None
    rotated = np.ones((3, 2, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(gui, "rotate_frame", lambda frame, rotation: calls.append((frame, rotation)) or rotated)

    assert annotator._read_frame(0)
    assert calls == [(original, -90)]
    assert annotator.current_frame is rotated


def test_session_checkpoint_persists_position_atomically_per_annotator(tmp_path) -> None:
    state = {
        "video_id": "sample-abc",
        "video_path": "clips/sample.mp4",
        "frame": 47,
        "playback_speed": 2.0,
    }
    jing_path = save_session(tmp_path, "Jing", state)
    assert jing_path == session_path(tmp_path, "Jing")
    assert not jing_path.with_suffix(".json.tmp").exists()
    assert load_session(tmp_path, "Jing")["frame"] == 47
    assert load_session(tmp_path, "someone else") is None


def test_corrupt_session_does_not_prevent_future_resume(tmp_path) -> None:
    path = session_path(tmp_path, "Jing")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{partial", encoding="utf-8")
    assert load_session(tmp_path, "Jing") is None
    save_session(tmp_path, "Jing", {"video_path": "videos/a.mov", "frame": 3})
    assert load_session(tmp_path, "Jing")["frame"] == 3


def test_gui_resumes_saved_video_frame_and_speed(tmp_path) -> None:
    from src.video_annotation import gui

    dataset = tmp_path / "dataset"
    videos_dir = dataset / "videos"
    videos_dir.mkdir(parents=True)
    video_paths = [videos_dir / "a.mov", videos_dir / "b.mov"]
    for path in video_paths:
        path.touch()
    config = load_config(DEFAULT_CONFIG)
    save_session(
        tmp_path / "annotations",
        "Jing",
        {"video_path": "videos/b.mov", "frame": 77, "playback_speed": 2.0},
    )
    annotator = gui.VideoAnnotator.__new__(gui.VideoAnnotator)
    annotator.video_paths = video_paths
    annotator.dataset_root = dataset
    annotator.output_dir = tmp_path / "annotations"
    annotator.annotator = "Jing"
    annotator.config = config
    annotator.speed_index = 0
    loaded = []

    def load_video(index):
        loaded.append(index)
        annotator.document = AnnotationDocument(
            VideoMetadata("b", "videos/b.mov", 10.0, 100), "Jing", "1.0"
        )
        annotator.frame_index = 0

    def read_frame(frame):
        annotator.frame_index = frame
        return True

    annotator._load_video = load_video
    annotator._read_frame = read_frame
    annotator._resume_session()

    assert loaded == [1]
    assert annotator.frame_index == 77
    assert config.playback_speeds[annotator.speed_index] == 2.0


def test_requested_navigation_shortcuts_are_configured() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.shortcuts["play_pause"] == " "
    assert config.shortcuts["previous_frame"] == "LEFT"
    assert config.shortcuts["next_frame"] == "RIGHT"
    assert config.shortcuts["speed_down"] == "DOWN"
    assert config.shortcuts["speed_up"] == "UP"
    assert config.shortcuts["previous_video"] == "-"
    assert config.shortcuts["next_video"] == "+"


def test_arrow_key_codes_are_normalized_across_opencv_backends() -> None:
    from src.video_annotation.gui import normalize_key_code

    assert normalize_key_code(65361) == "LEFT"       # Linux/X11
    assert normalize_key_code(2555904) == "RIGHT"   # Windows
    assert normalize_key_code(63232) == "UP"        # macOS
    assert normalize_key_code(16777237) == "DOWN"   # Qt
    assert normalize_key_code(ord("+")) == "+"
    assert normalize_key_code(-1) is None


def test_label_legend_uses_configured_colors() -> None:
    import numpy as np
    from src.video_annotation.gui import draw_label_legend

    config = load_config(DEFAULT_CONFIG)
    canvas = np.zeros((205, 900, 3), dtype=np.uint8)
    draw_label_legend(canvas, 0, config.labels)

    first_rgb = config.labels[0].color
    assert tuple(canvas[140, 14]) == tuple(reversed(first_rgb))
    assert np.count_nonzero(canvas) > 0


def test_numpad_label_mapping_matches_motion_layout() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.annotation_version == "1.1"
    assert config.labels_by_key["4"].id == "turning_left"
    assert config.labels_by_key["6"].id == "turning_right"
    assert config.labels_by_key["8"].id == "moving_forward"
    assert config.labels_by_key["2"].id == "moving_backward"
    assert config.labels_by_key["5"].id == "stationary"


def test_editing_a_frame_resets_all_later_keyframes() -> None:
    document = AnnotationDocument(metadata(), "tester", "1.1")
    document.set_keyframe(0, "stationary")
    document.set_keyframe(10, "moving_forward")
    document.set_keyframe(20, "turning_left")
    document.set_keyframe(40, "turning_right")

    document.set_keyframe(10, "moving_backward")
    cleared = document.clear_keyframes_after(10)

    assert cleared == 2
    assert document.keyframes == [
        Keyframe(0, "stationary"),
        Keyframe(10, "moving_backward"),
    ]
    assert document.segments()[-1].end_frame == 99
    assert document.segments()[-1].label == "moving_backward"


def test_committed_subset_is_fixed_unique_and_limited_to_100() -> None:
    import yaml

    payload = yaml.safe_load(DEFAULT_SUBSET.read_text(encoding="utf-8"))
    assert payload["seed"] == 20260807
    assert payload["count"] == 100
    assert len(payload["videos"]) == 100
    assert len(set(payload["videos"])) == 100
    assert "videos/0000f77c-6257be58.mov" in payload["videos"]


def test_subset_filter_preserves_manifest_order_and_rejects_missing(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    videos_dir = dataset / "videos"
    videos_dir.mkdir(parents=True)
    discovered = [videos_dir / "a.mov", videos_dir / "b.mov", videos_dir / "c.mov"]
    for path in discovered:
        path.touch()
    manifest = tmp_path / "subset.yaml"
    manifest.write_text(
        "count: 2\nvideos:\n  - videos/c.mov\n  - videos/a.mov\n", encoding="utf-8"
    )
    assert select_manifest_videos(discovered, dataset, manifest) == [discovered[2], discovered[0]]

    manifest.write_text("count: 1\nvideos: [videos/missing.mov]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        select_manifest_videos(discovered, dataset, manifest)


def test_rotation_is_inferred_per_video_from_prepared_frame(tmp_path) -> None:
    import cv2
    import numpy as np
    from src.video_annotation.orientation import infer_prepared_frame_rotation

    raw = np.zeros((120, 80, 3), dtype=np.uint8)
    raw[:40, :30] = (10, 40, 240)
    raw[55:110, 35:75] = (220, 180, 15)
    reference = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_dir = tmp_path / "frames" / "mixed-orientation"
    frame_dir.mkdir(parents=True)
    assert cv2.imwrite(str(frame_dir / "frame_00000.jpg"), reference)

    class Capture:
        def __init__(self):
            self.reset_to = None

        def read(self):
            return True, raw.copy()

        def set(self, _property, value):
            self.reset_to = value
            return True

    capture = Capture()
    rotation = infer_prepared_frame_rotation(capture, "videos/mixed-orientation.mov", tmp_path)

    assert rotation == 90
    assert capture.reset_to == 0
