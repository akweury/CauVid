# Local video segmentation annotator

A lightweight OpenCV desktop tool for labeling temporal segments in videos under `dataset/`. It never writes to or modifies video files. Each label press creates or changes a keyframe at the displayed frame. That label applies through the frame immediately before the next keyframe, and the final keyframe applies through the video's last frame.

## Install and launch

From the repository root, install the existing dependencies and launch the GUI:

```bash
pip install -r requirements.txt
python -m src.video_annotation annotate --annotator YOUR_NAME
```

The annotator is restricted to the fixed, ordered 100-video manifest at `configs/video_annotation/video_subset.yaml`; both annotation and validation use only those entries. The manifest records seed `20260807`, preserves existing annotation work, and prevents the order from changing between sessions. The default dataset is the same `config.get_dataset_path("driving_mini")` root used by `src/exp_july/pipeline.py`. It honors the project storage configuration, recursively scans its `videos/` directory, reads `configs/video_annotation/labels.yaml`, and atomically autosaves JSON files to `annotations/video_segmentation/`. Docker-style `/raw_driving_data` symlinks are mapped to the configured host-side `driving_raw` path. Override paths with `--dataset`, `--config`, or `--output`. Drag the **Frame** trackbar for frame-accurate seeking.

## Default shortcuts

| Key | Action |
|---|---|
| Numpad `4` / `6` | Label Left/Right |
| Numpad `8` / `2` | Label Forward/Backward |
| Numpad `5` | Label Stationary |
| Space | Play/pause |
| Left / Right | Previous/next frame |
| Down / Up | Slower/faster playback |
| `-` / `+` | Previous/next video |
| `z` / `c` | Previous/next labeled keyframe |
| `x` | Delete keyframe at the current frame |
| `q` | Quit |

All keys, labels, display names, RGB timeline colors, supported extensions, speeds, rotation handling, and the annotation version are configurable in `configs/video_annotation/labels.yaml`. By default, the GUI disables OpenCV autorotation and determines each video.s orientation by matching its raw first frame to the corresponding prepared pipeline frame. This supports mixed clockwise and counterclockwise clips. When no prepared reference is available, embedded rotation metadata is used; if that is also unavailable, `display.portrait_fallback_rotation: -90` rotates portrait frames clockwise to match the dataset annotations and prepared pipeline frames; set it to `null` to disable this fallback. Set `display.apply_rotation_metadata: false` only for videos whose pixels are already normalized. JSON configuration with the same fields is also supported. Keys must be unique; action shortcuts may be single printable characters or the named arrows `LEFT`, `RIGHT`, `UP`, and `DOWN`.

## Workflow

1. Open a video and seek to frame 0. Press its label key; a valid finished video must be labeled from frame 0 so it has no initial gap.
2. Play or seek to every point where the label changes and press the new label key. Assigning or changing a label clears every later keyframe in that video, so all later transitions must be reviewed and annotated again.
3. Review transitions with previous/next-keyframe shortcuts. Press another label key to change one, or `x` to delete it.
4. Move between videos. Every edit is immediately saved with an atomic replace and filesystem sync. A separate per-annotator session checkpoint records the current video, frame, and speed on navigation, every five seconds, and on exit; reopening with the same `--annotator` resumes that position.
5. Run final validation:

```bash
python -m src.video_annotation validate
```

Validation exits nonzero and reports missing/unlabeled videos, initial or internal gaps, duplicate keyframes, unknown labels, overlaps, and invalid frame ranges.

## Output schema

One JSON file is written per video. Paths are relative to the selected dataset root and IDs include a stable path hash to prevent filename collisions.

```json
{
  "schema_version": 1,
  "annotation_version": "1.0",
  "annotator": "alice",
  "updated_at": "2026-08-07T12:00:00+00:00",
  "video": {
    "id": "clips__drive-0123456789",
    "path": "clips/drive.mp4",
    "fps": 30.0,
    "frame_count": 900,
    "duration_seconds": 30.0
  },
  "keyframes": [
    {"frame": 0, "timestamp": 0.0, "label": "stationary"}
  ],
  "segments": [
    {
      "start_frame": 0,
      "end_frame": 899,
      "start_timestamp": 0.0,
      "end_timestamp": 29.966667,
      "label": "stationary"
    }
  ]
}
```

Frame ranges are inclusive. Timestamps identify the corresponding frame as `frame / fps`.
