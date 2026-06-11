# created on 02-03-2026
# Perception to Matrix
# Input: Video frames and corresponding bounding boxes, labels and dpeth maps
# Output: A matrix representation of the scene for each frame, including object positions, labels, and depth information
from copyreg import pickle

import pickle
import pathlib
import json
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
import torch 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from exp_driving_videos.modules.pipe_utils import exp_driving_utils as utils
from exp_driving_videos.modules.pipe_utils.visualization_pipeline import (
    trajectory_with_frames_visual,
    visualize_ego_rotation,
    visualize_ego_speed,
    visualize_full_scene,
    visualize_mask,
    visualize_obj_speed,
)
import config 


PIPELINE_DATA_FILENAME = "pipeline_data.pkl"
PIPELINE_DATA_VERSION = 1


def get_pipeline_data_file(out_path):
    return pathlib.Path(out_path) / PIPELINE_DATA_FILENAME


def load_pipeline_data(out_path, video_id=None):
    pipeline_file = get_pipeline_data_file(out_path)
    if pipeline_file.exists():
        with open(pipeline_file, "rb") as f:
            data = pickle.load(f)
        data.setdefault("schema_version", PIPELINE_DATA_VERSION)
        data.setdefault("video_id", video_id)
        data.setdefault("stages", {})
        return data

    return {
        "schema_version": PIPELINE_DATA_VERSION,
        "video_id": video_id,
        "stages": {},
    }


def save_pipeline_data(out_path, pipeline_data):
    pipeline_file = get_pipeline_data_file(out_path)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = pipeline_file.with_suffix(f"{pipeline_file.suffix}.tmp")
    with open(tmp_file, "wb") as f:
        pickle.dump(pipeline_data, f)
    os.replace(tmp_file, pipeline_file)
    return pipeline_file


def _stage_entry(value, params=None):
    return {
        "_stage_cache": True,
        "params": params or {},
        "data": value,
    }


def get_pipeline_stage(pipeline_data, stage_name, params=None):
    stages = pipeline_data.setdefault("stages", {})
    entry = stages.get(stage_name)
    if not entry:
        return None

    if isinstance(entry, dict) and entry.get("_stage_cache"):
        if params is None or (entry.get("params") or {}) == (params or {}):
            return entry.get("data")
        return None

    # Backward compatibility if an early consolidated file stored raw values.
    return entry if params in (None, {}) else None


def set_pipeline_stage(pipeline_data, out_path, stage_name, value, params=None):
    pipeline_data.setdefault("stages", {})[stage_name] = _stage_entry(value, params)
    return save_pipeline_data(out_path, pipeline_data)


def _legacy_load_pickle(file_path):
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _legacy_load_npz_dict(file_path):
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        return None
    loaded = np.load(file_path, allow_pickle=True)
    return {
        key: loaded[key].item()
        if loaded[key].shape == () and loaded[key].dtype == object
        else loaded[key]
        for key in loaded.files
    }


def _motion_param_subset(motion_cfg, keys):
    return {key: motion_cfg.get(key) for key in keys}


def _obj_matrix_params(cfg):
    if cfg is None:
        source_cfg = {}
    elif isinstance(cfg, (str, os.PathLike)) or "preprocess" in cfg:
        source_cfg = _get_preprocess_cfg(cfg)
    else:
        source_cfg = cfg
    return {
        "min_frame_number": source_cfg.get("min_frame_number", 5),
    }


def _filter_params(motion_cfg):
    return _motion_param_subset(motion_cfg, ["min_obj_frames", "min_bbox_area"])


def _ego_params(motion_cfg):
    return _motion_param_subset(
        motion_cfg,
        [
            "smooth_window",
            "stop_threshold",
            "min_stop_duration",
            "rotation_threshold",
            "min_rot_duration",
        ],
    )


def _object_speed_params(motion_cfg):
    return _motion_param_subset(motion_cfg, ["min_obj_frames", "min_bbox_area"])


def _get_detection_cfg(cfg=None):
    preprocess_cfg = _get_preprocess_cfg(cfg)
    det_cfg = {
        "enabled": True,
        "prepare_frame_adapter": False,
        "detection_output_root": str(config.get_output_path("pipeline_output") / "01_driving_mini_detection"),
        "strict": False,
    }
    det_cfg.update(preprocess_cfg.get("detection_annotations", {}))
    return det_cfg


def _load_detector_annotations(video_id, detection_cfg):
    if not detection_cfg.get("enabled", True):
        return {
            "enabled": False,
            "source": "driving_mini_detection",
            "video_id": video_id,
            "summary": {},
            "per_frame": {},
        }

    detection_root = pathlib.Path(detection_cfg.get("detection_output_root"))
    detections_file = detection_root / video_id / "detections.json"
    strict = bool(detection_cfg.get("strict", False))

    if not detections_file.exists():
        message = f"Detection annotations not found for {video_id}: {detections_file}"
        if strict:
            raise FileNotFoundError(message)
        print(f"[det-annotations] {message}")
        return {
            "enabled": True,
            "source": "driving_mini_detection",
            "video_id": video_id,
            "summary": {
                "available": False,
                "path": str(detections_file),
            },
            "per_frame": {},
        }

    with open(detections_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    per_frame = {}
    for frame_entry in raw.get("frames", []):
        frame_idx = int(frame_entry.get("frame_index", -1))
        if frame_idx < 0:
            continue
        per_frame[frame_idx] = {
            "boxes": frame_entry.get("boxes", []),
            "scores": frame_entry.get("scores", []),
            "labels": frame_entry.get("labels", []),
            "num_detections": frame_entry.get("num_detections", 0),
        }

    return {
        "enabled": True,
        "source": "driving_mini_detection",
        "video_id": raw.get("video_id", video_id),
        "summary": {
            "available": True,
            "path": str(detections_file),
            "num_frames": raw.get("num_frames", len(per_frame)),
            "num_detections": raw.get("num_detections", 0),
            "detected_classes": raw.get("detected_classes", {}),
        },
        "per_frame": per_frame,
    }


def _adapt_detector_annotations_to_frames(frames_data, detector_annotations):
    """
    Build a frame-aligned detector annotation view matching the frame timeline.

    This is an adapter-only interface for future integration and is intentionally
    not consumed by the current motion/perception stages.
    """
    per_frame = detector_annotations.get("per_frame", {})
    adapted = []

    for timeline_idx, frame_data in enumerate(frames_data):
        det_frame = per_frame.get(timeline_idx, {})
        adapted.append(
            {
                "timeline_index": timeline_idx,
                "frame": frame_data.get("frame"),
                "detector_boxes": det_frame.get("boxes", []),
                "detector_scores": det_frame.get("scores", []),
                "detector_labels": det_frame.get("labels", []),
                "detector_num_detections": det_frame.get("num_detections", 0),
            }
        )
    return adapted


def _load_or_compute_video_data(video_id, out_path, pipeline_data, cfg=None):
    detection_cfg = _get_detection_cfg(cfg)
    cached = get_pipeline_stage(pipeline_data, "video_data")
    if cached is not None:
        if "detector_annotations" not in cached:
            cached["detector_annotations"] = _load_detector_annotations(video_id, detection_cfg)
        if detection_cfg.get("prepare_frame_adapter", False) and "detector_annotations_frame_aligned" not in cached:
            cached["detector_annotations_frame_aligned"] = _adapt_detector_annotations_to_frames(
                cached.get("frames_data", []),
                cached.get("detector_annotations", {}),
            )
        if (not detection_cfg.get("prepare_frame_adapter", False)) and "detector_annotations_frame_aligned" in cached:
            # Keep cache lean when adapter output is disabled.
            cached.pop("detector_annotations_frame_aligned", None)
            set_pipeline_stage(pipeline_data, out_path, "video_data", cached)
        print(f"Loading cached video data from: {get_pipeline_data_file(out_path)}")
        return cached

    legacy_file = out_path / "video_data.pkl"
    legacy = _legacy_load_pickle(legacy_file)
    if legacy is not None:
        if "detector_annotations" not in legacy:
            legacy["detector_annotations"] = _load_detector_annotations(video_id, detection_cfg)
        if detection_cfg.get("prepare_frame_adapter", False):
            legacy["detector_annotations_frame_aligned"] = _adapt_detector_annotations_to_frames(
                legacy.get("frames_data", []),
                legacy.get("detector_annotations", {}),
            )
        print(f"Migrating cached video data into: {get_pipeline_data_file(out_path)}")
        set_pipeline_stage(pipeline_data, out_path, "video_data", legacy)
        return legacy

    input_data = utils.load_driving_mini_inputs(video_id)
    frames_data = utils.raw2frame_data(input_data, video_id)
    bg_mask = utils.extract_bg_mask(frames_data, video_id)
    depth_maps = [fd["depth_map"] for fd in frames_data]
    flows = _load_or_compute_flow(frames_data, video_id)
    detector_annotations = _load_detector_annotations(video_id, detection_cfg)
    video_data = {
        "frames_data": frames_data,
        "bg_masks": bg_mask,
        "depth_maps": depth_maps,
        "flows": flows,
        # Complementary external detector annotations (currently loaded only).
        "detector_annotations": detector_annotations,
    }
    if detection_cfg.get("prepare_frame_adapter", False):
        video_data["detector_annotations_frame_aligned"] = _adapt_detector_annotations_to_frames(
            frames_data,
            detector_annotations,
        )
    set_pipeline_stage(pipeline_data, out_path, "video_data", video_data)
    return video_data


def _load_or_compute_flow(frames_data, video_id):
    legacy_flow_file = config.get_output_path("pipeline_output") / f"optical_flow_{video_id}.npy"
    if legacy_flow_file.exists():
        print(f"Loading legacy cached optical flow: {legacy_flow_file}")
        return np.load(legacy_flow_file)

    flows = []
    for i in range(len(frames_data) - 1):
        frame1 = utils.load_frame(frames_data[i]["frame"])
        frame2 = utils.load_frame(frames_data[i + 1]["frame"])
        flow = utils.compute_optical_flow(frame1, frame2)
        flows.append(flow)
    return flows



def estimate3d_positions(bboxes, depth_map_file_name, frame_path):
    """
    Estimate 3D positions of objects based on bounding boxes and depth maps.

    Parameters:
    bboxes (list): A list of bounding boxes for the objects in the frame.
    depth_map_file_name (str): The file name of the depth map corresponding to the frame.
    frame_path (str): Path to the frame image to get its size.

    Returns:
    list: A list of estimated 3D positions for each object.
    """
    from PIL import Image
    import os
    
    def get_object_depth(depth, bbox):
        x1, y1, x2, y2 = bbox
        # Ensure bbox coordinates are integers and within bounds
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = depth[y1:y2, x1:x2]
        valid = crop[crop > 0]
        if len(valid) == 0:
            return torch.tensor(1.0)  # Default depth if no valid values
        return torch.median(valid)
    
    def backproject_center(bbox, depth, H, W):
        x1, y1, x2, y2 = bbox
        
        fx = fy = max(H, W)
        cx = W / 2.0
        cy = H / 2.0
        
        Z = get_object_depth(depth, bbox)
        
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        return torch.stack([X, Y, Z])
    
    # Load the frame image to get its size
    if os.path.isfile(frame_path):
        img = Image.open(frame_path)
        frame_height, frame_width = img.size[1], img.size[0]  # PIL: (width, height)
    else:
        # If frame_path is just a filename, we need to handle it differently
        # In this case, try to infer from depth map and scale if needed
        frame_height, frame_width = None, None
    
    # Load the depth map from the file with target size
    if frame_height is not None and frame_width is not None:
        depth_map = utils.load_depth_npz(depth_map_file_name, target_size=(frame_height, frame_width))
    else:
        depth_map = utils.load_depth_npz(depth_map_file_name)
    
    positions_3d = []
    
    for bbox in bboxes:
        H, W = depth_map.shape
        position_3d = backproject_center(bbox, depth_map, H, W)
        positions_3d.append(position_3d)
        
        
    return positions_3d



# def extract_bg_mask(frames_data, video_id=None):
    
#     bg_masks = []
#     for frame_data in frames_data:
#         frame = utils.load_frame(frame_data["frame"])
#         bboxes = frame_data["bboxes"]
#         labels = frame_data["labels"]
#         obj_ids = frame_data["obj_ids"]
        
#         # Create a background mask for the frame
#         bg_mask = utils.create_bg_mask(frame, bboxes, labels, obj_ids)
#         bg_masks.append(bg_mask)
#     return bg_masks



def raw2objs(frames_data, video_id, out_path, cfg, pipeline_data=None):
    
    """
    Convert raw frame data to objects with 3D positions
    Returns: a dict of objects, key is object id, value is a dict of object data, including label, position, frames, bboxes, frame_indices
    """
    
    
    params = _obj_matrix_params(cfg)
    if pipeline_data is not None:
        cached = get_pipeline_stage(pipeline_data, "smoothed_object_matrices", params)
        if cached is not None:
            print(f"Loading cached smoothed object matrices from: {get_pipeline_data_file(out_path)}")
            return cached

    matrix_file = out_path / f"smoothed_object_matrices.pkl"
    legacy = _legacy_load_pickle(matrix_file)
    if legacy is not None:
        print(f"Migrating cached smoothed object matrices into: {get_pipeline_data_file(out_path)}")
        if pipeline_data is not None:
            set_pipeline_stage(pipeline_data, out_path, "smoothed_object_matrices", legacy, params)
        return legacy

    obj_matrices = {}
    for frame_index, frame_data in enumerate(frames_data):
        # Process the frame data to create a matrix representation.
        bboxes = frame_data["bboxes"]
        labels = frame_data["labels"]
        depth_map = frame_data["depth_map"]
        frame = frame_data["frame"]
        obj_ids = frame_data["obj_ids"]
        
        positions_3d = estimate3d_positions(bboxes, depth_map, frame)  # Function to estimate 3D positions from bounding boxes and depth map
        for obj_id, label, position, bbox in zip(obj_ids, labels, positions_3d, bboxes):
            if obj_id not in obj_matrices:
                obj_matrices[obj_id] = {
                    "label": label,
                    "position": [position],
                    "frames": [frame],
                    "bboxes": [bbox],
                    "frame_indices": [frame_index]
                }
            else:
                obj_matrices[obj_id]["position"].append(position)
                obj_matrices[obj_id]["bboxes"].append(bbox)
                obj_matrices[obj_id]["frames"].append(frame)
                obj_matrices[obj_id]["frame_indices"].append(frame_index)
    # filter out objects that appear in less than min_frame_number frames
    min_frame_number = params["min_frame_number"]
    filtered_obj_matrices = {obj_id: data for obj_id, data in obj_matrices.items() if len(data["position"]) >= min_frame_number}

    if len(filtered_obj_matrices) == 0:
        print(f"No valid object matrices found for video {video_id} after filtering with min_frame_number={min_frame_number}.")
    smoothed_matrices = smooth_matrices(filtered_obj_matrices)
    if pipeline_data is not None:
        set_pipeline_stage(pipeline_data, out_path, "smoothed_object_matrices", smoothed_matrices, params)
    return smoothed_matrices






def smooth_matrices(obj_matrices):
    """
    Smooth the matrix representations of objects across frames.

    Parameters:
    obj_matrices (dict): A dictionary containing object matrices for each frame.

    Returns:
    dict: A dictionary containing smoothed object matrices.
    """
    def smooth_positions(positions_3d, window_size=5):
        #  window size is the number of frames to consider for smoothing
        smoothed_positions = []
        for i in range(len(positions_3d)):
            start = max(0, i - window_size // 2)
            end = min(len(positions_3d), i + window_size // 2 + 1)
            window_positions = positions_3d[start:end]
            smoothed_position = torch.mean(torch.stack(window_positions), dim=0)
            smoothed_positions.append(smoothed_position)
        return smoothed_positions

    smoothed_matrices = {}
    for obj_id, data in obj_matrices.items():
        label = data["label"]
        position = data["position"]
        position = [torch.tensor(p, dtype=torch.float32) for p in position]  # Convert positions to tensors
        smoothed_position = smooth_positions(position)  # Smooth the position across frames
        smoothed_matrices[obj_id] = {
            "label": label,
            "position": smoothed_position,
            "frames": data["frames"],
            "bboxes": data["bboxes"],
            "frame_indices": data["frame_indices"]
        }
    return smoothed_matrices

def save_matrices(matrices, output_path):
    """
    Save the smoothed object matrices to a file for later use.

    Parameters:
    matrices (dict): A dictionary containing smoothed object matrices.
    output_path (Path): Path to the file where matrices will be saved.
    """    
    ouput_file = output_path / f"smoothed_object_matrices.pkl"
    with open(ouput_file, 'wb') as f:
        pickle.dump(matrices, f)
    
    print(f"Saved smoothed object matrices to: {ouput_file}")
   

# def raw2objssss(frames_data,video_id, out_path):
#     # load the matrix if it already exists
#     matrix_file = out_path / f"smoothed_object_matrices.pkl"
#     if matrix_file.exists():
#         print(f"Matrix file already exists for video {video_id}, loading from file: {matrix_file}")
#         smoothed_matrices = utils.load_matrix(matrix_file)  # This will print the loaded matrix for debugging
#     else:
#         obj_matrices = raw2objs(frames_data, video_id)     
#         smoothed_matrices = smooth_matrices(obj_matrices)
#         # save the smoothed_matrices for later use
#         save_matrices(smoothed_matrices, out_path)    
#     return smoothed_matrices


def _compute_obj_speed_from_flow(bboxes, frame_indices, bg_masks, flows, depth_maps, ego_motion,
                                  min_bbox_area=400):
    ego_vx, ego_vz = ego_motion
    if len(bboxes) == 0:
        return [],[]
    if len(bboxes) == 1:
        return [0.0],[0.0]

    vx_obj_speeds = []
    vz_obj_speeds = []
    for i in range(1, len(bboxes)):
        f_prev = frame_indices[i - 1]
        f_curr = frame_indices[i]
        # skip observation if bbox is too small
        b = bboxes[i]
        x1, y1, x2, y2 = (b.tolist() if hasattr(b, "tolist") else list(b))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < min_bbox_area:
            vx_obj_speeds.append(float('nan'))
            vz_obj_speeds.append(float('nan'))
            continue
        vx_obj, vz_obj, moving = utils.estimate_obj_velo(flows[i], depth_maps[i], bg_masks[i], bboxes[i], ego_vx[i], ego_vz[i])
        vx_obj_speeds.append(vx_obj)
        vz_obj_speeds.append(vz_obj)

    # first observation has no predecessor; mirror observation 1
    vx_obj_speeds = [vx_obj_speeds[0]] + vx_obj_speeds
    vz_obj_speeds = [vz_obj_speeds[0]] + vz_obj_speeds
    

    
    return vx_obj_speeds, vz_obj_speeds

def est_obj_rel_speed(obj_speeds, ego_data):
    """
    Convert object speeds from ego-centric frame to world frame by adding the
    ego vehicle's own speed at each frame.

    Parameters
    ----------
    obj_speeds : dict
        {obj_id: np.ndarray of shape (num_frames, 2)}
        Each row is [vx, vz]; rows equal to -1.0 indicate the object was absent.
    ego_data : dict
        Must contain "ego_x_speeds" and "ego_z_speeds" keys — per-frame ego speed lists as
        returned by percept2ego_speed (length == num_frames).

    Returns
    -------
    obj_speeds_rel : dict
        Same structure as obj_speeds; absent frames remain -1.0.
    """
    ego_vx = list(ego_data["ego_x_speeds"])
    ego_vz = list(ego_data["ego_z_speeds"])

    obj_speeds_rel = {}
    for obj_id, speed_track in obj_speeds.items():
        rel_track = speed_track.copy()
        num_frames = rel_track.shape[0]
        for fi in range(num_frames):
            vx, vz = rel_track[fi]
            if vx == -1.0 and vz == -1.0:  # absent sentinel
                continue
            ex = ego_vx[fi] if fi < len(ego_vx) else 0.0
            ez = ego_vz[fi] if fi < len(ego_vz) else 0.0
            rel_track[fi] = [vx + ex, vz + ez]
        obj_speeds_rel[obj_id] = rel_track
    return obj_speeds_rel

def percept2obj_dist_rank_and_bboxes(obj_matrix, video_data):
    """For each object, compute its distance rank among all tracked objects per frame
    (rank 1 = closest to ego), and return the per-frame bboxes.

    Returns
    -------
    obj_ranks : dict  {obj_id: list[int]}
        Length == num_frames; -1 where the object is absent, otherwise 1-based
        rank (1 = closest, 2 = second-closest, …).
    obj_bboxes : dict  {obj_id: {fi: list}}
        Sparse dict of frame_index → bbox for each object.
    """
    frames_data = video_data["frames_data"]
    num_frames = len(frames_data)

    # ── pass 1: collect raw z-depth and bboxes per object per frame ──────────
    obj_depths  = {}   # {obj_id: {fi: float}}
    obj_bboxes  = {}   # {obj_id: {fi: list}}
    for obj_id, data in obj_matrix.items():
        positions     = data["position"]
        frame_indices = data["frame_indices"]
        bboxes        = data["bboxes"]
        fi_depth_map  = {}
        fi_bbox_map   = {}
        for pos, fi, bb in zip(positions, frame_indices, bboxes):
            if fi < num_frames:
                fi_depth_map[fi] = pos[2].item()   # z-coord as proxy for ego distance
                fi_bbox_map[fi]  = bb.tolist() if hasattr(bb, "tolist") else list(bb)
        obj_depths[obj_id] = fi_depth_map
        obj_bboxes[obj_id] = fi_bbox_map

    # ── pass 2: rank objects by depth within each frame ───────────────────────
    # For each frame, sort present objects by ascending z (smaller z = closer).
    obj_ranks = {obj_id: [-1] * num_frames for obj_id in obj_matrix}
    for fi in range(num_frames):
        present = [(oid, obj_depths[oid][fi])
                   for oid in obj_depths if fi in obj_depths[oid]]
        if not present:
            continue
        # sort ascending by depth; assign rank 1 to closest
        present.sort(key=lambda x: x[1])
        for rank, (oid, _) in enumerate(present, start=1):
            obj_ranks[oid][fi] = rank

    return obj_ranks, obj_bboxes
    
    
    
    
def percept2obj_speed(video_id, obj_matrix, video_data, ego_data, cfg=None, pipeline_data=None):
    """
    Compute per-frame speed for every tracked object across the full video.

    For each object the 3D displacement between consecutive observed frames is
    used as the speed estimate.  Frames where the object is absent are filled
    with -1.0.

    Parameters
    ----------
    obj_matrix : dict
        Smoothed object matrices as returned by smooth_matrices / percept2matrix.
        Each entry: {obj_id: {"position": list[Tensor(3,)],
                               "frame_indices": list[int], ...}}
    merged_mask : list[int]
        Per-frame state mask (0=driving, 1=stopped, 2=turning).
        Its length determines num_frames.

    Returns
    -------
    obj_speeds : dict
        {obj_id: list[float]}  length == num_frames.
        -1.0 indicates the object was not observed at that frame.
    """
    
    bg_masks = video_data["bg_masks"]
    depth_maps = video_data["depth_maps"]
    flows = video_data["flows"]
    
    min_obj_frames = cfg.get("min_obj_frames", 10) if cfg else 10
    min_bbox_area  = cfg.get("min_bbox_area", 400) if cfg else 400
    
    ego_motion = (ego_data["ego_x_speeds"], ego_data["ego_z_speeds"])
    params = _object_speed_params(cfg or {})
    out_path = cfg.get("out_path") if cfg else None
    if pipeline_data is not None:
        cached = get_pipeline_stage(pipeline_data, "object_speeds", params)
        if cached is not None:
            print(f"Loading cached object speeds from: {get_pipeline_data_file(out_path)}")
            return cached

    obj_speed_file = config.get_output_path("pipeline_output") / f"object_speeds_{video_id}.pkl"
    legacy = _legacy_load_pickle(obj_speed_file)
    if legacy is not None:
        if out_path is not None:
            print(f"Migrating cached object speeds into: {get_pipeline_data_file(out_path)}")
        else:
            print(f"Loading legacy cached object speeds: {obj_speed_file}")
        if pipeline_data is not None and out_path is not None:
            set_pipeline_stage(pipeline_data, out_path, "object_speeds", legacy, params)
        return legacy

    num_frames = len(bg_masks)
    obj_speeds = {}

    for obj_id, data in obj_matrix.items():
        positions     = data["position"]        # list of Tensor(3,)
        frame_indices = data["frame_indices"]   # list of int
        bboxes = data["bboxes"]

        # skip objects that appear too rarely
        if len(frame_indices) < min_obj_frames:
            continue

        # core speed computation over observed frames only
        obj_vx, obj_vz = _compute_obj_speed_from_flow(bboxes, frame_indices, bg_masks, flows, depth_maps, ego_motion, min_bbox_area)
        # scatter into the full timeline; NaN entries (small-bbox frames) stay absent
        speed_track = np.zeros((num_frames, 2), dtype=np.float32) - 1.0
        for obs_i, fi in enumerate(frame_indices):
            if 0 <= fi < num_frames:
                vx, vz = obj_vx[obs_i], obj_vz[obs_i]
                if not (np.isnan(vx) or np.isnan(vz)):
                    speed_track[fi] = [vx, vz]
        obj_speeds[obj_id] = speed_track

    if pipeline_data is not None and out_path is not None:
        set_pipeline_stage(pipeline_data, out_path, "object_speeds", obj_speeds, params)
    return obj_speeds

def estimate_obj_z_motion(obj_speeds, motion_cfg):
    """
    Return segment-level motion stats instead of frame labels.
    """
    obj_motion = {}
    for obj_id, speed_track in obj_speeds.items():
        vz = np.array([float(s[1]) for s in speed_track])
        
        valid_mask = vz != -1 
        vz = vz[valid_mask]
        
        if len(vz) == 0:
            obj_motion[obj_id] = -1
            continue
        
        # adaptive thresholds
        eps = 0.1 * np.std(vz) if np.std(vz) > 0 else 0.01
        
        # -- ratios --
        approach_ratio = (vz<-eps).mean()
        away_ratio = (vz>eps).mean()
        
        # -- mean / strength ---
        vz_mean = np.mean(vz)
        vz_std = np.std(vz)
        
        # --- change detection ---
        sign = np.sign(vz)
        change_ratio = (np.diff(sign) != 0).mean() if len(sign) > 1 else 0
        
        # final label
        if approach_ratio > 0.7:
            label = "approaching"
        elif away_ratio > 0.7:
            label = "moving_away"
        else:
            label = "transition"
            
        obj_motion[obj_id] = {
            "label": label,
            "approach_ratio": approach_ratio,
            "away_ratio": away_ratio,
            "vz_mean": vz_mean,
            "vz_std": vz_std,
            "change_ratio": change_ratio
        }
    return obj_motion 
    # def _classify(s):
    #     if float(s[0]) == -1.0 and float(s[1]) == -1.0:
    #         return -1
    #     vz = float(s[1])
    #     if vz < -approach_threshold:
    #         return 1   # approaching
    #     if vz > away_threshold:
    #         return 2   # moving away
    #     return 0       # same distance

    # obj_motion = {}
    # for obj_id, speed_track in obj_speeds.items():
    #     raw = [_classify(s) for s in speed_track]

    #     # collapse short approaching/away runs back to "same"
    #     result = raw[:]
    #     i = 0
    #     while i < len(raw):
    #         if raw[i] in (1, 2):
    #             cat = raw[i]
    #             j = i
    #             while j < len(raw) and raw[j] == cat:
    #                 j += 1
    #             if j - i < min_run_duration:
    #                 for k in range(i, j):
    #                     result[k] = 0
    #             i = j
    #         else:
    #             i += 1

    #     obj_motion[obj_id] = result

    # return obj_motion


def estimate_obj_x_motion(obj_speeds, motion_cfg):
    """
    Similar to estimate_obj_z_motion but for lateral (x) motion.
    """
    obj_motion = {}
    for obj_id, speed_track in obj_speeds.items():
        vx = np.array([float(s[0]) for s in speed_track])
        
        valid_mask = vx != -1 
        vx = vx[valid_mask]
        
        if len(vx) == 0:
            obj_motion[obj_id] = -1
            continue
        
        eps = 0.1 * np.std(vx) if np.std(vx) > 0 else 0.01
        
        left_ratio = (vx<-eps).mean()
        right_ratio = (vx>eps).mean()
        
        if left_ratio > 0.7:
            label = "left"
        elif right_ratio > 0.7:
            label = "right"
        else:
            label = "stable"
            
        obj_motion[obj_id] = {
            "label": label,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "vx_mean": np.mean(vx),
            "vx_std": np.std(vx)
        }
    return obj_motion

def percept2ego_speed(video_data, cfg):
    """
    Estimate per-frame ego speed from optical flow of the background.

    Parameters
    ----------
    video_id : str
    save_flow : bool
        Cache the computed optical flow to disk so subsequent calls skip RAFT.
    smooth_window : int
        Half-width of the moving-average smoothing window applied to raw speeds.
        Set to 1 to disable smoothing.

    Returns
    -------
    ego_speeds_smoothed : list of float
        Smoothed per-frame speed values, length == number of frames.
    """
    smooth_window = cfg.get("smooth_window", 5)
    flows      = video_data.get("flows", [])
    bg_masks   = video_data.get("bg_masks", [])
    depth_maps  = video_data.get("depth_maps", [])
    ego_x_speeds,ego_z_speeds = [], []
    for i in range(len(flows)):
        ego_x_speed, ego_z_speed = utils.estimate_ego_motion(flows[i], bg_masks[i], utils.load_depth_npz(depth_maps[i]))
        ego_x_speeds.append(ego_x_speed)
        ego_z_speeds.append(ego_z_speed)
    # frame 0 has no preceding flow; mirror frame 1
    if len(ego_x_speeds) > 1:
        ego_x_speeds[0] = ego_x_speeds[1]
        ego_z_speeds[0] = ego_z_speeds[1]

    ego_x_speeds = [float(ego_x_speeds[0])] + [float(s) for s in ego_x_speeds]
    ego_z_speeds = [float(ego_z_speeds[0])] + [float(s) for s in ego_z_speeds]
    
    # moving-average smoothing
    ego_x_speeds_smoothed = []
    for i in range(len(ego_x_speeds)):
        start = max(0, i - smooth_window // 2)
        end   = min(len(ego_x_speeds), i + smooth_window // 2 + 1)
        ego_x_speeds_smoothed.append(float(np.mean(ego_x_speeds[start:end])))

    ego_z_speeds_smoothed = []
    for i in range(len(ego_z_speeds)):
        start = max(0, i - smooth_window // 2)
        end   = min(len(ego_z_speeds), i + smooth_window // 2 + 1)
        ego_z_speeds_smoothed.append(float(np.mean(ego_z_speeds[start:end])))
    
    return ego_x_speeds_smoothed, ego_z_speeds_smoothed


def percept2ego_rotation(bg_masks, flows, depth_maps, rotation_mask_cfg):
    smooth_window = rotation_mask_cfg.get("smooth_window", 5)
    ego_rotations = []
    for i in range(len(flows)):
        ego_rotation = utils.estimate_ego_rotation(flows[i], bg_masks[i], utils.load_depth_npz(depth_maps[i]))
        ego_rotations.append(ego_rotation)
    # frame 0 has no preceding flow; mirror frame 1
    if len(ego_rotations) > 1:
        ego_rotations[0] = ego_rotations[1]


    ego_rotations = [ego_rotations[0]] + [float(r) for r in ego_rotations]  # align with frame 0
    # moving-average smoothing
    ego_rotations_smoothed = []
    for i in range(len(ego_rotations)):
        start = max(0, i - smooth_window // 2)
        end   = min(len(ego_rotations), i + smooth_window // 2 + 1)
        ego_rotations_smoothed.append(float(np.mean(ego_rotations[start:end])))

    return ego_rotations_smoothed


def estimate_ego_stop(ego_x_speeds,ego_z_speeds, cfg):
    """
    Classify each frame as stopped or driving based on smoothed ego speed.
    Stop runs shorter than *min_stop_duration* frames are discarded.

    Parameters
    ----------
    ego_x_speeds : list of float
        Per-frame smoothed x-axis speeds as returned by percept2ego_speed (aligned to
        frame indices, i.e. length == num_frames).
    ego_z_speeds : list of float
        Per-frame smoothed z-axis speeds as returned by percept2ego_speed (aligned to
        frame indices, i.e. length == num_frames).
    stop_threshold : float
        Frames whose |speed| is strictly below this value are labelled stopped.
    min_stop_duration : int
        Minimum number of consecutive stopped frames required to keep a stop
        region.  Shorter runs are relabelled as driving.

    Returns
    -------
    stopped : list of bool
        True  → ego is stopped at that frame.
        False → ego is driving.
    """
    stop_threshold = cfg.get("stop_threshold", 0.01)
    min_stop_duration = cfg.get("min_stop_duration", 3)

    ego_speed = [np.sqrt(x**2 + z**2) for x, z in zip(ego_x_speeds, ego_z_speeds)]
    raw = [s < stop_threshold for s in ego_speed]

    # filter out stop runs shorter than min_stop_duration
    result = raw[:]
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i < min_stop_duration:
                for k in range(i, j):
                    result[k] = False
            i = j
        else:
            i += 1

    return result

def estimate_ego_rotation(ego_rotations, rotation_mask_cfg):
    """
    Classify each frame as turning or straight based on smoothed ego rotation.
    Rotation runs shorter than *min_rot_duration* frames are discarded.

    Parameters
    ----------
    ego_rotations : list of float
        Per-frame smoothed rotations as returned by percept2ego_rotation (aligned to
        frame indices, i.e. length == num_frames).
    rotation_threshold : float
        Frames whose |rotation| is strictly above this value are labelled turning.
    min_rot_duration : int
        Minimum number of consecutive turning frames required to keep a turn
        region.  Shorter runs are relabelled as straight.

    Returns
    -------
    turning : list of bool
        True  → ego is turning at that frame.
        False → ego is going straight.
    """
    rotation_threshold = rotation_mask_cfg.get("rotation_threshold", 0.1)
    min_rot_duration = rotation_mask_cfg.get("min_rot_duration", 3)
    
    raw = [abs(r) > rotation_threshold for r in ego_rotations]

    # filter out turn runs shorter than min_rot_duration
    result = raw[:]
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i < min_rot_duration:
                for k in range(i, j):
                    result[k] = False
            i = j
        else:
            i += 1

    return result

def est_obj_motion_data(video_data, ego_data, obj_matrix, video_id, motion_cfg, vis_cfg, pipeline_data=None):
    
    out_path = motion_cfg.get("out_path", get_out_path(video_id))
    params = _object_speed_params(motion_cfg)
    if pipeline_data is not None:
        cached = get_pipeline_stage(pipeline_data, "obj_motion_data", params)
        if cached is not None:
            print(f"Loading cached object motion data from: {get_pipeline_data_file(out_path)}")
            return cached

    obj_motion_data_file = out_path / f"obj_motion_data.npz"
    legacy = _legacy_load_npz_dict(obj_motion_data_file)
    if legacy is not None:
        print(f"Migrating cached object motion data into: {get_pipeline_data_file(out_path)}")
        if pipeline_data is not None:
            set_pipeline_stage(pipeline_data, out_path, "obj_motion_data", legacy, params)
        return legacy
    
    
    obj_speeds = percept2obj_speed(video_id, obj_matrix, video_data, ego_data, cfg=motion_cfg, pipeline_data=pipeline_data)
    obj_ranks, obj_bboxes = percept2obj_dist_rank_and_bboxes(obj_matrix, video_data)    
    obj_data = {
        "obj_speeds": obj_speeds,
        "obj_ranks": obj_ranks,
        "obj_bboxes": obj_bboxes
    }
    # visualize_full_scene(video_data["frames_data"], video_id, ego_data, obj_data, out_path)
    # visualize_obj_speed(video_data["frames_data"], video_id, ego_data, obj_data, **vis_cfg)

    if pipeline_data is not None:
        set_pipeline_stage(pipeline_data, out_path, "obj_motion_data", obj_data, params)
        print(f"Saved object motion data for video {video_id} → {get_pipeline_data_file(out_path)}")
    return obj_data
    
def get_out_path(video_id):
    out_path = config.get_output_path("pipeline_output") / f"{video_id}"
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def est_motion_mask(video_id, video_data, motion_cfg, vis_cfg, pipeline_data=None):
       
    out_path = get_out_path(video_id)
    params = _ego_params(motion_cfg)
    if pipeline_data is not None:
        cached = get_pipeline_stage(pipeline_data, "ego_data", params)
        if cached is not None:
            print(f"Loading cached ego motion data from: {get_pipeline_data_file(out_path)}")
            return cached

    stop_mask_file = out_path / f"stop_mask.npy"
    turnning_mask_file = out_path / f"turning_mask.npy"
    ego_x_speeds_file = out_path / f"ego_x_speeds.npy"
    ego_z_speeds_file = out_path / f"ego_z_speeds.npy"
    
    if stop_mask_file.exists() and ego_x_speeds_file.exists() and ego_z_speeds_file.exists() and turnning_mask_file.exists():
        print(f"Stop mask file already exists for video {video_id}, loading from file: {stop_mask_file}")
        stopped_mask = np.load(stop_mask_file)
        turning_mask = np.load(turnning_mask_file)
        ego_x_speeds = np.load(ego_x_speeds_file)
        ego_z_speeds = np.load(ego_z_speeds_file)
        ego_data = {
            "ego_x_speeds": ego_x_speeds,
            "ego_z_speeds": ego_z_speeds,
            "stopped_mask": stopped_mask,
            "turning_mask": turning_mask,
        }
        if pipeline_data is not None:
            print(f"Migrating cached ego motion data into: {get_pipeline_data_file(out_path)}")
            set_pipeline_stage(pipeline_data, out_path, "ego_data", ego_data, params)
        return ego_data
    else:
        """Estimate and visualize the ego stop mask for a single video."""
        ego_x_speeds, ego_z_speeds = percept2ego_speed(video_data, motion_cfg)
        stopped_mask = estimate_ego_stop(ego_x_speeds,ego_z_speeds, motion_cfg)
        turning_mask = estimate_ego_rotation(ego_x_speeds, motion_cfg)
        
        print(f"[stop] Visualizing ego speed for video: {video_id}")
        visualize_mask(video_data['frames_data'], ego_z_speeds, ego_x_speeds, stopped_mask,turning_mask, video_id, **vis_cfg)
        # visualize_ego_speed(frames_data, ego_x_speeds, ego_z_speeds, video_id, stopped_mask,turning_mask, **vis_cfg)

    ego_data = {
        "ego_x_speeds": ego_x_speeds,
        "ego_z_speeds": ego_z_speeds,
        "stopped_mask": stopped_mask,
        "turning_mask": turning_mask,
    }
    if pipeline_data is not None:
        set_pipeline_stage(pipeline_data, out_path, "ego_data", ego_data, params)
    return ego_data 


def est_turn_mask(video_id,frames_data, bg_masks,flows,depth_maps, rotation_mask_cfg, vis_cfg):
    
    turn_mask_file = config.get_output_path("pipeline_output") / f"{video_id}_turn_mask.npy"
    ego_rotation_file = config.get_output_path("pipeline_output") / f"{video_id}_ego_rotations.npy"
    if turn_mask_file.exists() and ego_rotation_file.exists():
        print(f"Turn mask file already exists for video {video_id}, loading from file: {turn_mask_file}")
        turn_mask = np.load(turn_mask_file)
        ego_rotations = np.load(ego_rotation_file)
    else:
        """Estimate and visualize the ego turn mask for a single video."""
        ego_rotations = percept2ego_rotation(bg_masks, flows, depth_maps, rotation_mask_cfg)
        turn_mask = estimate_ego_rotation(ego_rotations, rotation_mask_cfg)
        print(f"[turn] Visualizing ego rotation for video: {video_id}")
        visualize_ego_rotation(frames_data, ego_rotations, video_id, turn_mask=turn_mask, **vis_cfg)
        # save the turn mask and ego rotations for later use
        np.save(turn_mask_file, turn_mask)
        np.save(ego_rotation_file, ego_rotations)
    return turn_mask, ego_rotations

def merge_segs(stopped_mask, turn_mask):
    # Create a merged mask with 3 classes: 0=driving, 1=stopped, 2=turning
    merged_mask = []
    for stop, turn in zip(stopped_mask, turn_mask):
        if stop:
            merged_mask.append(1)  # stopped
        elif turn:
            merged_mask.append(2)  # turning
        else:
            merged_mask.append(0)  # driving
            
    
    return merged_mask
def rank_objects_by_distance(obj_matrix, depth_maps, frames_data):
    """
    Rank objects by their median distance to the ego across all observed frames.

    Parameters
    ----------
    obj_matrix : dict
        {obj_id: {"label": str,
                  "position":     list[Tensor(3,)],
                  "frames":       list[str],
                  "bboxes":       list,
                  "frame_indices": list[int]}}
    depth_maps : list of str
        File paths to per-frame depth maps, aligned to frame indices.
    frames_data : list of dict
        As returned by raw2frame_data — one dict per frame with a "frame" key.

    Returns
    -------
    dict  — same structure as input obj_matrix but with an added "distance_rank" key
            for each object, where 0 = closest object by median distance, 1 = second closest, etc.
    """
    ranked = {}
    for obj_id, data in obj_matrix.items():
        positions     = data["position"]
        frames        = data["frames"]
        bboxes        = data["bboxes"]
        frame_indices = data["frame_indices"]
        label         = data["label"]

        distances = []
        for pos, fi in zip(positions, frame_indices):
            depth_map_path = depth_maps[fi]
            depth_map = utils.load_depth_npz(depth_map_path)
            x, y, z = pos.tolist() if hasattr(pos, "tolist") else list(pos)
            # guard against out-of-bounds coordinates
            h, w = depth_map.shape
            px = min(max(int(round(x)), 0), w - 1)
            py = min(max(int(round(y)), 0), h - 1)
            distance = float(depth_map[py, px])
            distances.append(distance)

        median_distance = np.median(distances) if distances else float("inf")
        ranked[obj_id] = {
            "label": label,
            "position": positions,
            "frames": frames,
            "bboxes": bboxes,
            "frame_indices": frame_indices,
            "median_distance": median_distance,
        }

    # assign distance ranks (0=closest)
    sorted_objs = sorted(ranked.items(), key=lambda item: item[1]["median_distance"])
    for rank, (obj_id, data) in enumerate(sorted_objs):
        ranked[obj_id]["distance_rank"] = rank

    return ranked

def filter_obj_matrix(obj_matrix, motion_cfg, pipeline_data=None):
    """
    Filter an object matrix, keeping only objects with sufficient observations
    and reasonably sized bounding boxes.

    Two stages:
      1. Per-frame pruning: remove individual observations whose bbox area is
         smaller than *min_bbox_area* (pixels²).  Positions, frames, bboxes,
         and frame_indices are all trimmed together.
      2. Object pruning: drop any object whose remaining observation count is
         below *min_obj_frames*.

    Parameters
    ----------
    obj_matrix : dict
        {obj_id: {"label": str,
                  "position":     list[Tensor(3,)],
                  "frames":       list[str],
                  "bboxes":       list,
                  "frame_indices": list[int]}}
    min_obj_frames : int
        Minimum number of valid observations required to keep an object.
    min_bbox_area : float
        Observations with bbox area < this value are discarded.

    Returns
    -------
    dict  — filtered subset of obj_matrix with the same structure.
    """
    out_path = motion_cfg.get("out_path", config.get_output_path("pipeline_output"))
    min_obj_frames = motion_cfg.get("min_obj_frames", 5)
    min_bbox_area = motion_cfg.get("min_bbox_area", 400)
    params = _filter_params(motion_cfg)
    if pipeline_data is not None:
        cached = get_pipeline_stage(pipeline_data, "filtered_obj_matrix", params)
        if cached is not None:
            print(f"Loading cached filtered object matrix from: {get_pipeline_data_file(out_path)}")
            return cached
    
    filtered_file = out_path / f"filtered_obj_matrix_minframes{min_obj_frames}_minarea{min_bbox_area}.pkl"
    legacy = _legacy_load_pickle(filtered_file)
    if legacy is not None:
        print(f"Migrating cached filtered object matrix into: {get_pipeline_data_file(out_path)}")
        if pipeline_data is not None:
            set_pipeline_stage(pipeline_data, out_path, "filtered_obj_matrix", legacy, params)
        return legacy
    else:
        filtered = {}
        for obj_id, data in obj_matrix.items():
            positions     = data["position"]
            frames        = data["frames"]
            bboxes        = data["bboxes"]
            frame_indices = data["frame_indices"]
            label         = data["label"]

            # ── stage 1: drop observations with tiny bboxes ──────────────────────
            keep_pos, keep_fr, keep_bb, keep_fi = [], [], [], []
            for pos, frm, bb, fi in zip(positions, frames, bboxes, frame_indices):
                coords = bb.tolist() if hasattr(bb, "tolist") else list(bb)
                x1, y1, x2, y2 = coords
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area >= min_bbox_area:
                    keep_pos.append(pos)
                    keep_fr.append(frm)
                    keep_bb.append(bb)
                    keep_fi.append(fi)

            # ── stage 2: drop objects with too few valid observations ─────────────
            if len(keep_fi) < min_obj_frames:
                continue

            filtered[obj_id] = {
                "label":        label,
                "position":     keep_pos,
                "frames":       keep_fr,
                "bboxes":       keep_bb,
                "frame_indices": keep_fi,
            }

        print(f"filter_obj_matrix: {len(obj_matrix)} objects → {len(filtered)} kept "
            f"(min_frames={min_obj_frames}, min_bbox_area={min_bbox_area})")
        if pipeline_data is not None:
            set_pipeline_stage(pipeline_data, out_path, "filtered_obj_matrix", filtered, params)
        return filtered


def _get_preprocess_cfg(cfg=None):
    if cfg is None or isinstance(cfg, (str, os.PathLike)):
        return utils.get_pattern_cfg(cfg).get("preprocess", {})
    if "preprocess" in cfg:
        return cfg.get("preprocess", {})
    return cfg

def _get_cfgs(out_path, cfg=None):
    preprocess_cfg = _get_preprocess_cfg(cfg)
    motion_cfg = {
        "save_flow":         True,
        "smooth_window":     5,
        "stop_threshold":    10.0,  # m/s  (ego stop detection)
        "min_stop_duration": 3,
        "rotation_threshold": 25,
        "min_rot_duration":   3,
        "min_obj_frames":    10,   # skip objects seen in fewer frames
        "min_bbox_area":     5000,  # skip frames where bbox area < this (px²)
        "approach_threshold": 50,  # vz < -this → approaching
        "away_threshold":     50,  # vz >  this  → moving away
        "left_threshold":     50,  # vx < -this → moving left
        "right_threshold":    50,  # vx >  this  → moving right
        "min_run_duration":   3,   # min consecutive frames to keep approach/away/left/right label
        "out_path":          out_path,
    }
    motion_cfg.update(preprocess_cfg.get("motion", {}))
    motion_cfg["out_path"] = out_path

    vis_cfg = {
        "fps":          5,
        "chart_width":  10,
        "chart_height": 3,
        "output_dir":   out_path
    }
    vis_cfg.update(preprocess_cfg.get("visualization", {}))
    vis_cfg["output_dir"] = out_path
    detection_cfg = _get_detection_cfg(cfg)
    return motion_cfg, vis_cfg, detection_cfg
    
    
    
def run_single_video(video_id, cfg=None):
    out_path = get_out_path(video_id)
    pipeline_data = load_pipeline_data(out_path, video_id)
    motion_cfg, vis_cfg, detection_cfg = _get_cfgs(out_path, cfg)
    
    # ----------- Estimate ego motion masks (stopped, turning) and visualize them -----------
    print(f"Estimating Ego mask for video: {video_id}")
    video_data = _load_or_compute_video_data(video_id, out_path, pipeline_data, cfg={"preprocess": {"detection_annotations": detection_cfg}})
    det_summary = video_data.get("detector_annotations", {}).get("summary", {})
    if det_summary.get("available", False):
        print(
            "Loaded complementary detector annotations: "
            f"{det_summary.get('num_frames', 0)} frames, "
            f"{det_summary.get('num_detections', 0)} detections"
        )
    ego_data = est_motion_mask(video_id, video_data, motion_cfg, vis_cfg, pipeline_data)

    # ----------- Estimate object motion data and visualize the full scene -----------
    print(f"Estimating Other mask for video: {video_id}")
    objs = raw2objs(video_data['frames_data'], video_id, out_path, cfg, pipeline_data)
    objs_filtered = filter_obj_matrix(objs, motion_cfg, pipeline_data)

    ranked_params = _filter_params(motion_cfg)
    objs_ranked = get_pipeline_stage(pipeline_data, "ranked_obj_matrix", ranked_params)
    if objs_ranked is None:
        objs_ranked = rank_objects_by_distance(objs_filtered, video_data['depth_maps'], video_data['frames_data'])
        set_pipeline_stage(pipeline_data, out_path, "ranked_obj_matrix", objs_ranked, ranked_params)
    else:
        print(f"Loading cached ranked object matrix from: {get_pipeline_data_file(out_path)}")
    
    # ------------ Estimate object motion data and visualize the full scene -----------
    obj_data = est_obj_motion_data(video_data, ego_data, objs_ranked, video_id, motion_cfg, vis_cfg, pipeline_data)
    print(f"Finished processing video {video_id}. Output saved to {get_pipeline_data_file(out_path)}")
    
    
def main():
    video_ids = config.get_mini_video_ids()
    for video_id in video_ids:
        print(f"\n=== Processing video {video_id} ===")
        run_single_video(video_id)
        
        
if __name__ == "__main__":
    main()
    
