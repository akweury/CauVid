import json
import os
import sys
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
from pathlib import Path

import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.exp_driving_videos import pipeline_config as driving_pipeline_config
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import ego_motion_driving_mini
from src.exp_driving_videos.modules import merge_gt_and_detected_driving_mini
from src.exp_driving_videos.modules import prepare_3d_positions_driving_mini
from src.exp_driving_videos.modules import tracking_driving_mini
from tqdm import tqdm


def get_pipeline_output_root():
    return Path(os.environ.get("CAUVID_PIPELINE_OUTPUT_PATH", ROOT / "output_july"))


def normalize_detection_image_paths(video_result, dataset_root):
    video_id = str(video_result.get("video_id", "")).strip()
    if not video_id:
        return video_result, False

    frames_root = Path(dataset_root) / "frames" / video_id
    changed = False
    updated = dict(video_result)
    updated_frames = []
    for frame in video_result.get("frames", []):
        frame_record = dict(frame)
        image_path_text = str(frame_record.get("image_path", "")).strip()
        image_path = Path(image_path_text) if image_path_text else None
        if image_path_text and image_path and not image_path.exists():
            candidate = frames_root / image_path.name
            if candidate.exists():
                frame_record["image_path"] = str(candidate)
                changed = True
        updated_frames.append(frame_record)
    updated["frames"] = updated_frames
    return updated, changed


def write_detection_cache_if_needed(video_result, source_path=None):
    detections_json = str(video_result.get("output_paths", {}).get("detections_json", "")).strip()
    path = Path(source_path) if source_path is not None else (Path(detections_json) if detections_json else None)
    if path is None:
        return
    updated = dict(video_result)
    output_paths = dict(updated.get("output_paths", {}))
    output_paths["detections_json"] = str(path)
    updated["output_paths"] = output_paths
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_step6_cache_payload(payload, video_id):
    return (
        isinstance(payload, dict)
        and str(payload.get("video_id", "")) == str(video_id)
        and isinstance(payload.get("frames", []), list)
    )


def collect_track_lengths(tracking_results):
    lengths = []
    for video_result in tracking_results:
        summaries = video_result.get("accepted_tracks", {}).get("track_summaries", [])
        for summary in summaries:
            lengths.append(int(summary.get("track_length", 0)))
    return lengths


def track_length_range_counts(track_lengths):
    ranges = [
        ("1-4", 1, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20-49", 20, 49),
        ("50-99", 50, 99),
        ("100+", 100, None),
    ]
    counts = []
    for label, start, end in ranges:
        if end is None:
            count = sum(1 for length in track_lengths if length >= start)
        else:
            count = sum(1 for length in track_lengths if start <= length <= end)
        counts.append((label, count))
    return counts


def save_track_length_histogram(track_lengths, output_root):
    if not track_lengths:
        return None
    track_count = len(track_lengths)
    figure_path = Path(output_root) / f"track_length_histogram_n{track_count}.png"
    plt.figure(figsize=(8, 4.5))
    plt.hist(track_lengths, bins=20, color="#4C78A8", edgecolor="black")
    plt.xlabel("Track length")
    plt.ylabel("Number of tracks")
    plt.yscale("log")
    plt.title(f"Track Length Distribution (n={track_count})")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def step1_init(video_ids=None, video_count=None):
    dataset_root = config.get_dataset_path("driving_mini")
    video_dir = dataset_root / "videos"
    all_videos = sorted(config.get_mini_video_ids()) if video_dir.exists() else []
    if video_ids:
        videos = []
        for video_id in video_ids:
            if video_id and video_id not in videos:
                videos.append(video_id)
    else:
        videos = list(all_videos)
    if video_count is not None:
        videos = videos[:video_count]
    print(
        f"[step 1] loaded {len(videos)} videos for this run \n"
        f"[step 1] from {video_dir} \n"
        f"[step 1] (dataset=driving_mini, total_in_dataset={len(all_videos)})"
    )
    detection_args = {
        "video_ids": videos,
        "model_name": driving_pipeline_config.DRIVING_MINI_OD_MODEL,
        "classes": driving_pipeline_config.DRIVING_MINI_OD_CLASSES,
        "output_root": get_pipeline_output_root() / "01_driving_mini_detection",
        "od_calibration_policy": {},
        "force_recompute": False,
        "render_video": driving_pipeline_config.get_detection_render_video_enabled(default=True),
        "check_cache": driving_pipeline_config.get_detection_check_cache_enabled(default=False),
        "enable_candidate_branch": driving_pipeline_config.get_detection_candidate_branch_enabled(default=False),
        "skip_step": driving_pipeline_config.get_detection_skip_step_enabled(default=False),
    }
    tracking_args = {
        "output_root": get_pipeline_output_root() / "02_driving_mini_tracking",
        "frame_rate": 10,
        "tracker_args": None,
        "force_recompute": False,
        "render_video": driving_pipeline_config.get_tracking_render_video_enabled(default=True),
    }
    positions_3d_args = {
        "output_root": get_pipeline_output_root() / "06_driving_mini_3d_positions",
        "model_name": "depth-anything/DA3-Large",
        "batch_size": 4,
        "device": "auto",
        "force_recompute": False,
        "force_recompute_depth": False,
    }
    ego_motion_args = {
        "output_root": get_pipeline_output_root() / "07_driving_mini_ego_motion",
        "force_recompute": False,
        "smoothing_window": driving_pipeline_config.get_ego_motion_smoothing_window(default=5),
        "static_adjust_cfg": driving_pipeline_config.get_ego_static_adjustment_cfg(),
        "render_video": driving_pipeline_config.get_ego_motion_render_video_enabled(default=True),
        "flow_device": "auto",
    }
    return {
        "videos": videos,
        "dataset_root": dataset_root,
        "detection_args": detection_args,
        "tracking_args": tracking_args,
        "positions_3d_args": positions_3d_args,
        "ego_motion_args": ego_motion_args,
    }


def step2_detection(env, args):
    videos = env["videos"]
    if not videos:
        print("[step 2] no videos selected, skip detection")
        return {"videos": [], "detections": [], "detection_output_root": None}

    run_args = dict(args)
    model_name = run_args["model_name"]
    classes = run_args["classes"]
    skip_step = bool(run_args.pop("skip_step", False))
    render_video = bool(run_args.get("render_video", True))
    check_cache = bool(run_args.get("check_cache", False))
    candidate_branch_enabled = bool(run_args.get("enable_candidate_branch", False))
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[step 2] model={model_name}")
    print(f"[step 2] classes={len(classes)} render_video={render_video} check_cache={check_cache}")
    print(f"[step 2] output_root={output_root}")

    if skip_step:
        manifest_path = output_root / "detection_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        video_entries = {
            str(entry.get("video_id", "")).strip(): dict(entry)
            for entry in manifest.get("videos", [])
            if str(entry.get("video_id", "")).strip()
        }
        detections = []
        for video_id in videos:
            entry = video_entries[video_id]
            detections_path = Path(str(entry.get("detections_json", "")).strip() or output_root / video_id / "detections.json")
            with detections_path.open("r", encoding="utf-8") as f:
                video_result = json.load(f)
            if hasattr(detect_driving_mini, "_apply_candidate_branch_mode"):
                video_result = detect_driving_mini._apply_candidate_branch_mode(video_result, candidate_branch_enabled)
            detections.append(video_result)
        print(f"[step 2] loaded cached detection results for {len(detections)} videos")
    else:
        detections = detect_driving_mini.run(**run_args)
        print(f"[step 2] completed detection for {len(detections)} videos")

    accepted_count = sum(int(video_result.get("num_detections", 0)) for video_result in detections)
    candidate_count = sum(int(video_result.get("num_candidate_detections", 0)) for video_result in detections)
    normalized_detections = []
    rewritten_count = 0
    for video_result in detections:
        normalized_video_result, changed = normalize_detection_image_paths(video_result, env["dataset_root"])
        if changed:
            rewritten_count += 1
            write_detection_cache_if_needed(
                normalized_video_result,
                source_path=output_root / str(normalized_video_result.get("video_id", "")).strip() / "detections.json",
            )
        normalized_detections.append(normalized_video_result)
    detections = normalized_detections
    if rewritten_count:
        print(f"[step 2] rewrote frame paths for {rewritten_count} cached detection files")
    print(f"[step 2] accepted_detections={accepted_count}, candidate_detections={candidate_count}")
    return {
        "videos": videos,
        "detections": detections,
        "detection_output_root": output_root,
        "model_name": model_name,
        "classes": classes,
        "tracking_args": env["tracking_args"],
        "positions_3d_args": env["positions_3d_args"],
        "ego_motion_args": env["ego_motion_args"],
    }


def step3_tracking(detection_state):
    videos = detection_state["videos"]
    detections = detection_state["detections"]
    if not videos or not detections:
        print("[step 3] no detection results, skip tracking")
        return {"videos": videos, "tracks": [], "tracking_output_root": None}

    run_args = dict(detection_state["tracking_args"])
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    render_video = bool(run_args.get("render_video", True))
    if render_video and tracking_driving_mini.cv2 is None:
        print("[step 3][warn] OpenCV is unavailable; tracking rendering disabled")
        render_video = False
    tracking_driving_mini.ensure_tracking_runtime_available()

    tracking_results = []
    progress = tqdm(detections, desc="[step 3] tracking", unit="video")
    for video_result in progress:
        progress.set_postfix_str(str(video_result.get("video_id", "")), refresh=False)
        tracking_results.append(
            tracking_driving_mini.track_video(
                video_result=video_result,
                output_root=output_root,
                frame_rate=int(run_args.get("frame_rate", 10)),
                tracker_args=run_args.get("tracker_args"),
                force_recompute=bool(run_args.get("force_recompute", False)),
                render_video=render_video,
            )
        )

    manifest = {
        "schema_version": getattr(tracking_driving_mini, "_TRACKING_SCHEMA_VERSION", 7),
        "candidate_branch_enabled": all(bool(r.get("candidate_branch_enabled", True)) for r in tracking_results),
        "num_videos": len(tracking_results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
                "num_tracking_input_candidate_detections": int(r.get("num_tracking_input_candidate_detections", 0)),
                "num_candidate_tracks": int(r.get("num_candidate_tracks", 0)),
                "num_raw_candidate_tracks": int(r.get("num_raw_candidate_tracks", 0)),
                "num_deduplicated_candidate_tracks": int(r.get("num_deduplicated_candidate_tracks", 0)),
                "num_rejected_candidate_tracks": int(r.get("num_rejected_candidate_tracks", 0)),
                "num_rejected_candidate_detections": int(r.get("num_rejected_candidate_detections", 0)),
            }
            for r in tracking_results
        ],
    }
    with (output_root / "tracks_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    track_lengths = collect_track_lengths(tracking_results)
    for label, count in track_length_range_counts(track_lengths):
        print(f"[step 3] track_length_range {label}: {count}")
    figure_path = save_track_length_histogram(track_lengths, output_root)
    print(
        f"[step 3] done videos={len(tracking_results)} "
        f"tracks={sum(int(row.get('num_tracks', 0)) for row in tracking_results)}"
    )
    if figure_path is not None:
        print(f"[step 3] histogram={figure_path}")
    return {
        "videos": videos,
        "tracks": tracking_results,
        "tracking_output_root": output_root,
        "positions_3d_args": detection_state["positions_3d_args"],
        "ego_motion_args": detection_state["ego_motion_args"],
    }

def step6_positions_3d(tracking_state):
    videos = tracking_state["videos"]
    tracking_results = tracking_state["tracks"]
    if not videos or not tracking_results:
        print("[step 6] no tracking results, skip 3d positions")
        return {"videos": videos, "positions_3d": [], "positions_3d_output_root": None}

    run_args = dict(tracking_state["positions_3d_args"])
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[step 6] output_root={output_root}")
    merged_results = [
        merge_gt_and_detected_driving_mini._tracked_video_as_merged_result(video_result)
        for video_result in tracking_results
    ]
    positions_3d = []
    cached_videos = 0
    progress = tqdm(merged_results, desc="[step 6] positions_3d", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        cache_path = output_root / video_id / "positions_3d.json"
        cached_result = None
        if not bool(run_args.get("force_recompute", False)):
            payload = load_json_if_exists(cache_path)
            if is_step6_cache_payload(payload, video_id):
                cached_result = payload
        if cached_result is not None:
            cached_videos += 1
            positions_3d.append(cached_result)
            continue
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            positions_3d.append(
                prepare_3d_positions_driving_mini.process_video(
                    video_result=video_result,
                    output_root=output_root,
                    model_name=run_args.get("model_name", "depth-anything/DA3-Large"),
                    batch_size=int(run_args.get("batch_size", 4)),
                    device=str(run_args.get("device", "auto")),
                    force_recompute=bool(run_args.get("force_recompute", False)),
                    force_recompute_depth=bool(run_args.get("force_recompute_depth", False)),
                )
            )
    manifest = {
        "version": getattr(prepare_3d_positions_driving_mini, "_POSITIONS_3D_VERSION", 4),
        "model_name": run_args.get("model_name", "depth-anything/DA3-Large"),
        "num_videos": len(positions_3d),
        "num_frames_total": sum(r.get("num_frames", 0) for r in positions_3d),
        "num_objects_with_3d_total": sum(r.get("num_objects_with_3d", 0) for r in positions_3d),
        "num_candidate_objects_with_3d_total": sum(r.get("num_candidate_objects_with_3d", 0) for r in positions_3d),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r.get("num_frames", 0),
                "num_objects_with_3d": r.get("num_objects_with_3d", 0),
                "num_candidate_objects_with_3d": r.get("num_candidate_objects_with_3d", 0),
                "depth_dir": r.get("depth_dir", ""),
            }
            for r in positions_3d
        ],
    }
    with (output_root / "positions_3d_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 6] done videos={len(positions_3d)} "
        f"cached={cached_videos} "
        f"objects_with_3d={sum(int(row.get('num_objects_with_3d', 0)) for row in positions_3d)} "
        f"candidate_objects_with_3d={sum(int(row.get('num_candidate_objects_with_3d', 0)) for row in positions_3d)}"
    )
    return {
        "videos": videos,
        "positions_3d": positions_3d,
        "positions_3d_output_root": output_root,
        "ego_motion_args": tracking_state["ego_motion_args"],
    }


def step7_ego_motion(position_state):
    videos = position_state["videos"]
    positions_3d = position_state["positions_3d"]
    if not videos or not positions_3d:
        print("[step 7] no 3d positions, skip ego motion")
        return {"videos": videos, "ego_motion": [], "ego_motion_output_root": None}

    run_args = dict(position_state["ego_motion_args"])
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    ego_motion = []
    cached_videos = 0
    progress = tqdm(positions_3d, desc="[step 7] ego_motion", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        cache_path = output_root / video_id / "ego_motion.json"
        cached_result = None
        if not bool(run_args.get("force_recompute", False)):
            payload = load_json_if_exists(cache_path)
            if (
                payload
                and int(payload.get("version", 0)) == getattr(ego_motion_driving_mini, "_EGO_MOTION_VERSION", 0)
                and str(payload.get("estimation_method", "")) == getattr(ego_motion_driving_mini, "_EGO_MOTION_METHOD", "")
            ):
                cached_result = payload
        if cached_result is not None:
            cached_videos += 1
            ego_motion.append(cached_result)
            continue
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ego_motion.append(
                ego_motion_driving_mini.process_video(
                    video_result=video_result,
                    output_root=output_root,
                    force_recompute=bool(run_args.get("force_recompute", False)),
                    smoothing_window=int(run_args.get("smoothing_window", 5)),
                    static_adjust_cfg=run_args.get("static_adjust_cfg"),
                    render_video=bool(run_args.get("render_video", True)),
                    flow_device=run_args.get("flow_device"),
                )
            )
    manifest = {
        "num_videos": len(ego_motion),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_frames_with_ego_motion": r["num_frames_with_ego_motion"],
            }
            for r in ego_motion
        ],
    }
    with (output_root / "ego_motion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 7] done videos={len(ego_motion)} "
        f"cached={cached_videos} "
        f"frames_with_ego_motion={sum(int(row.get('num_frames_with_ego_motion', 0)) for row in ego_motion)}"
    )
    return {
        "videos": videos,
        "ego_motion": ego_motion,
        "ego_motion_output_root": output_root,
    }


def step7b_tracklet_repair(position_state, ego_state):
    # 1. perform only the safest interpolations
    # handle short-term omissions, 
    # such as missing few frames, 
    # where the preceding and 
    # following bounding boxes are contiguous, 
    # the class is consistent, and motion is smooth. 
    # In this round, interpolated bounding boxes 
    # can be generated directly with a high degree 
    # of confidence.
    
    
    
    return {
        "videos": position_state["videos"],
        "tracklet_repair": [],
        "positions_3d": position_state.get("positions_3d", []),
        "positions_3d_output_root": position_state.get("positions_3d_output_root"),
        "ego_motion": ego_state["ego_motion"],
        "ego_motion_output_root": ego_state.get("ego_motion_output_root"),
    }


def step8_relative_object_motion(position_state, repaired_state):
    return {"videos": position_state["videos"], "relative_object_motion": []}


def step9_temporal_segmentation(ego_state, relative_motion_state):
    return {"videos": ego_state["videos"], "temporal_segments": []}


def step10_segment_object_motion(segment_state):
    return {"videos": segment_state["videos"], "segment_object_motion": []}
