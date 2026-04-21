# created on 02-03-2026
# Perception to Matrix
# Input: Video frames and corresponding bounding boxes, labels and dpeth maps
# Output: A matrix representation of the scene for each frame, including object positions, labels, and depth information
from copyreg import pickle

import pickle
import pathlib
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
import torch 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.exp_driving_videos.pipe_utils import exp_driving_utils as utils
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


def _load_or_compute_video_data(video_id, out_path, pipeline_data):
    cached = get_pipeline_stage(pipeline_data, "video_data")
    if cached is not None:
        print(f"Loading cached video data from: {get_pipeline_data_file(out_path)}")
        return cached

    legacy_file = out_path / "video_data.pkl"
    legacy = _legacy_load_pickle(legacy_file)
    if legacy is not None:
        print(f"Migrating cached video data into: {get_pipeline_data_file(out_path)}")
        set_pipeline_stage(pipeline_data, out_path, "video_data", legacy)
        return legacy

    input_data = utils.load_driving_mini_inputs(video_id)
    frames_data = utils.raw2frame_data(input_data, video_id)
    bg_mask = utils.extract_bg_mask(frames_data, video_id)
    depth_maps = [fd["depth_map"] for fd in frames_data]
    flows = _load_or_compute_flow(frames_data, video_id)
    video_data = {
        "frames_data": frames_data,
        "bg_masks": bg_mask,
        "depth_maps": depth_maps,
        "flows": flows,
    }
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

def trajectory_with_frames_visual(matrices, smooth_matrices, input_data, output_path):
    os.makedirs(output_path, exist_ok=True)
    # for each trajectory, show the smoothed trajectory on the left, highlight the current position, 
    # and the image on the right,bounding box the the target object, saved as a gif file
    
    def create_trajectory_figure_with_current(position, current_idx, obj_id, label, target_height, title_suffix=""):
        """
        Create trajectory plot with current position highlighted.
        
        Args:
            position: Full trajectory positions (tensor)
            current_idx: Index of current frame to highlight
            obj_id: Object ID
            label: Object label
            target_height: Target height in pixels for the output image
        
        Returns:
            Trajectory image as numpy array (RGB)
        """
        position = torch.stack(position)  # Ensure position is a tensor of shape (num_frames, 3)
        # Create the trajectory plot
        fig, ax = plt.subplots(figsize=(5, 5))
        x = position[:, 0].numpy()
        z = position[:, 2].numpy()
        
        # Plot full trajectory
        ax.plot(x, z, marker='o', linestyle='-', linewidth=2, markersize=4, color='gray', alpha=0.5)
        
        # Highlight trajectory up to current frame
        if current_idx > 0:
            ax.plot(x[:current_idx+1], z[:current_idx+1], marker='o', linestyle='-', 
                   linewidth=2, markersize=4, color='blue', label='Trajectory')
        
        # Add start marker
        ax.scatter(x[0], z[0], color='green', s=150, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
        
        # Add current position marker (larger and distinct)
        ax.scatter(x[current_idx], z[current_idx], color='red', s=200, marker='*', 
                  label='Current', zorder=10, edgecolors='black', linewidths=2)
        
        # Add end marker
        ax.scatter(x[-1], z[-1], color='orange', s=150, marker='X', label='End', zorder=5, edgecolors='black', linewidths=2)
        
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Z (meters)", fontsize=12)
        ax.set_title(f"Object {obj_id} ({label})\nFrame {current_idx+1}/{len(position)} {title_suffix}", 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        trajectory_img = buf[:, :, :3]  # Convert RGBA to RGB
        plt.close(fig)
        
        # Resize to match target height
        trajectory_height, trajectory_width = trajectory_img.shape[:2]
        new_width = int(trajectory_width * target_height / trajectory_height)
        trajectory_img_resized = cv2.resize(trajectory_img, (new_width, target_height))
        
        return trajectory_img_resized
        
        
    for obj_id, data in matrices.items():
        label = data["label"]
        position = data["position"]
        frames = data["frames"]
        bboxes = data["bboxes"]
        
        # each bbox is one frame, so we can visualize the trajectory and the frame with bbox side by side
        visual_frames = []
        for i, (bbox, frame) in enumerate(zip(bboxes, frames)):
            frame_img = utils.load_frame(frame, bbox, obj_id, label)  # Load the frame and draw the bounding box
            frame_height = frame_img.shape[0]
            
            # Create trajectory plot with current position highlighted
            trajectory_img_current = create_trajectory_figure_with_current(
                position, i, obj_id, label, frame_height
            )
            
            # Create smoothed trajectory plot with current position highlighted
            trajectory_img_current_smoothed = create_trajectory_figure_with_current(
                smooth_matrices[obj_id]["position"], i, obj_id, label, frame_height, title_suffix="(Smoothed)"
                
            )
            
            # Combine trajectory and frame side by side
            combined_img = np.hstack((trajectory_img_current, 
                                      trajectory_img_current_smoothed, 
                                      frame_img))
            visual_frames.append(combined_img)
        
        # save the visual frames as a gif, with 2 fps
        safe_label = label.replace(' ', '_').replace('/', '_')
        gif_path = output_path / f"obj_{obj_id}_{safe_label}_{len(bboxes)}frames.gif"
        imageio.mimsave(gif_path, visual_frames, fps=2)
        print(f"Saved trajectory gif for Object {obj_id} ({label}): {gif_path}")

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

def visualize_full_scene(frames_data, video_id, ego_data, obj_data, out_path, fps=5):
    """
    Produce a single MP4 showing every frame with ALL tracked objects annotated.

    Per frame:
      - Rank-1 (closest) object: bright-green bbox, thicker border, "Rank #1" label
      - All other present objects: cyan-yellow bbox, normal border, "Rank #N | <label>" label
      - Above every object bbox: object motion-mask state, e.g. "Rel: Approaching | Left"
      - Bottom-left: ego motion overlay (Moving/Stopped, Straight/Turning)

    Output: *out_path*/full_scene.mp4
    """
    out_file = out_path / f"full_scene.mp4"
    # if video is exist then return
    
    # if out_file.exists():
    #     print(f"Full-scene video already exists for video {video_id}, skipping generation: {out_file}")
    #     return
    
    
    

    obj_bboxes = obj_data.get("obj_bboxes", {})
    obj_ranks  = obj_data.get("obj_ranks",  {})
    obj_z_motion = obj_data.get("motion_vz")
    obj_x_motion = obj_data.get("motion_vx")

    _Z_LABEL = {
        -1: ("Absent",      (80,  80,  80)),
         0: ("Same Dist.",  (0,  200,   0)),
         1: ("Approaching", (0,   60, 220)),
         2: ("Moving Away", (220, 80,   0)),
    }
    _X_LABEL = {
        -1: ("Absent",  (80,  80,  80)),
         0: ("Stable",  (180, 180,   0)),
         1: ("Left",    (255, 100, 100)),
         2: ("Right",   (100, 100, 255)),
    }

    ego_vx_full      = list(ego_data["ego_x_speeds"])
    ego_vz_full      = list(ego_data["ego_z_speeds"])
    num_frames       = len(frames_data)
    ego_stopped_full = list(ego_data.get("stopped_mask", [False] * num_frames))
    ego_turn_full    = list(ego_data.get("turn_mask",    [False] * num_frames))
    for _sig in (ego_vx_full, ego_vz_full, ego_stopped_full, ego_turn_full):
        if len(_sig) < num_frames:
            _sig += [_sig[-1]] * (num_frames - len(_sig))
    ego_stopped_full = [bool(v) for v in ego_stopped_full[:num_frames]]
    ego_turn_full    = [bool(v) for v in ego_turn_full[:num_frames]]

    # precompute per-object label lookup {obj_id: {fi: label_str}}
    obj_label_map = {}
    for fi, frame_data in enumerate(frames_data):
        obj_ids_fi = frame_data.get("obj_ids", [])
        labels_fi  = frame_data.get("labels", [])
        for k, oid in enumerate(obj_ids_fi):
            if oid not in obj_label_map:
                obj_label_map[oid] = {}
            obj_label_map[oid][fi] = labels_fi[k] if k < len(labels_fi) else str(oid)



    first_img     = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_file), fourcc, fps, (frame_w, frame_h))

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, frame_w / 1000)
    thickness  = max(2, int(font_scale * 2))
    pad        = int(10 * font_scale)

    # colour constants
    _COLOR_RANK1   = (0, 255, 0)        # bright green  – closest object
    _COLOR_OTHER   = (0, 220, 220)      # cyan-yellow   – all other objects
    _THICK_RANK1   = thickness * 2
    _THICK_OTHER   = thickness

    def _motion_text_and_color(oid, fi):
        """Return a concise object motion-mask row for the full-scene overlay."""
        prefix = "Rel"
        z_track = obj_z_motion.get(oid) if obj_z_motion is not None else None
        x_track = obj_x_motion.get(oid) if obj_x_motion is not None else None
        if z_track is None and x_track is None:
            return None, (200, 200, 200)

        z_cat = z_track[fi] if z_track is not None and fi < len(z_track) else -1
        x_cat = x_track[fi] if x_track is not None and fi < len(x_track) else -1
        z_text, z_color = _Z_LABEL.get(z_cat, ("Unknown", (128, 128, 128)))
        x_text, x_color = _X_LABEL.get(x_cat, ("Unknown", (128, 128, 128)))
        mask_color = x_color if x_cat in (1, 2) else z_color
        return f"{prefix}: {z_text} | {x_text}", mask_color

    def _draw_label_rows(img, x, y_top, rows):
        row_gap = max(2, int(3 * font_scale))
        line_sizes = [cv2.getTextSize(text, font, scale, thick) for text, scale, thick, _ in rows]
        total_h = sum(sz[0][1] + sz[1] for sz in line_sizes) + row_gap * (len(rows) - 1)
        max_w = max(sz[0][0] for sz in line_sizes)

        text_x = min(max(0, x), max(0, frame_w - max_w - 8))
        first_baseline = max(y_top - 4, total_h + 4)
        y_positions = []
        cursor = first_baseline
        for idx in range(len(rows) - 1, -1, -1):
            (_tw, th), baseline = line_sizes[idx]
            y_positions.insert(0, cursor)
            cursor -= th + baseline + row_gap

        bg_x1 = max(0, text_x - 4)
        bg_x2 = min(frame_w, text_x + max_w + 8)
        bg_y1 = max(0, y_positions[0] - line_sizes[0][0][1] - 3)
        bg_y2 = min(frame_h, y_positions[-1] + line_sizes[-1][1] + 3)
        label_ov = img.copy()
        cv2.rectangle(label_ov, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(label_ov, 0.4, img, 0.6, 0, img)

        for (text, scale, thick, color), y_pos in zip(rows, y_positions):
            cv2.putText(img, text, (text_x, y_pos), font, scale, color, thick, cv2.LINE_AA)

    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)

        # ── draw all tracked objects ──────────────────────────────────────────
        for oid, fi_bbox_map in obj_bboxes.items():
            bb = fi_bbox_map.get(fi)
            if bb is None:
                continue
            bb = bb.tolist() if hasattr(bb, "tolist") else list(bb)
            ox1, oy1, ox2, oy2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

            rank_track = obj_ranks.get(oid, [])
            rank_val   = rank_track[fi] if fi < len(rank_track) else -1

            is_closest = (rank_val == 1)
            color      = _COLOR_RANK1 if is_closest else _COLOR_OTHER
            bthick     = _THICK_RANK1 if is_closest else _THICK_OTHER

            cv2.rectangle(frame_bgr, (ox1, oy1), (ox2, oy2), color, bthick)

            rank_str  = f"Rank #{rank_val}" if rank_val >= 1 else "Rank: ?"
            obj_label = obj_label_map.get(oid, {}).get(fi, str(oid))
            title_str = f"{rank_str} | {obj_label}"
            motion_str, motion_color = _motion_text_and_color(oid, fi)

            # draw label rows just above the bbox
            label_rows = [(title_str, font_scale * 0.8, bthick, color)]
            if motion_str:
                label_rows.append((motion_str, font_scale * 0.7, _THICK_OTHER, motion_color))
            _draw_label_rows(frame_bgr, ox1, oy1, label_rows)

        # ── ego state label — bottom-left ─────────────────────────────────────
        ego_z_text  = "Stopped" if ego_stopped_full[fi] else "Moving"
        ego_z_color = (255, 165, 0)   if ego_stopped_full[fi] else (200, 200, 200)
        ego_x_text  = "Turning" if ego_turn_full[fi] else "Straight"
        ego_x_color = (180, 100, 220) if ego_turn_full[fi] else (200, 200, 200)

        e_lines = [("Ego", (255, 255, 255)), (ego_z_text, ego_z_color), (ego_x_text, ego_x_color)]
        e_sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t, _ in e_lines]
        ebox_w  = max(sz[0][0] for sz in e_sizes) + 3 * pad
        ebox_h  = sum(sz[0][1] + sz[1] for sz in e_sizes) + (len(e_lines) + 1) * pad
        eby0    = frame_h - ebox_h - pad
        ego_ov  = frame_bgr.copy()
        cv2.rectangle(ego_ov, (pad, eby0), (pad + ebox_w, eby0 + ebox_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(ego_ov, 0.45, frame_bgr, 0.55, 0, frame_bgr)
        ey = eby0
        for (etxt, ecol), ((_etw, _eth), _ebl) in zip(e_lines, e_sizes):
            ey += _eth + pad
            cv2.putText(frame_bgr, etxt, (2 * pad, ey), font, font_scale, ecol, thickness, cv2.LINE_AA)
            ey += _ebl

        writer.write(frame_bgr)

    writer.release()
    print(f"Saved full-scene visualization ({num_frames} frames) → {out_file}")


def visualize_obj_speed(frames_data, video_id, ego_data, obj_motion_data,
                        fps=5, chart_width=10, chart_height=3, output_dir=None):
    """
    Produce one MP4 per tracked object with the video frame on top and four
    signal subplots stacked below:
      1. Object vz (forward)  — Approaching (steelblue) / Moving Away (tomato)
      2. Object vx (lateral)  — Left (cornflowerblue)   / Right (salmon)
      3. Ego    vz (forward)  — plain speed signal
      4. Ego    vx (lateral)  — plain speed signal

    State labels for object z-motion and x-motion are overlaid top-left on the
    frame (z-motion on line 1, x-motion on line 2).

    Z-motion categories  (-1/0/1/2): absent / same-dist / approaching / moving-away
    X-motion categories  (-1/0/1/2): absent / stable    / left        / right

    Output: *output_dir*/<video_id>_obj_<obj_id>_speed.mp4
    """
    import pathlib

    # z-motion: category → (display text, BGR colour)
    _Z_LABEL = {
        -1: ("Absent",      (80,  80,  80)),
         0: ("Same Dist.",  (0,  200,   0)),
         1: ("Approaching", (0,   60, 220)),
         2: ("Moving Away", (220, 80,   0)),
    }
    # x-motion: category → (display text, BGR colour)
    _X_LABEL = {
        -1: ("Absent",  (80,  80,  80)),
         0: ("Stable",  (180, 180,   0)),
         1: ("Left",    (255, 100, 100)),
         2: ("Right",   (100, 100, 255)),
    }
    ego_motion = (ego_data["ego_x_speeds"], ego_data["ego_z_speeds"])
    obj_speeds = obj_motion_data.get("obj_speeds", {})
    obj_z_motion_abs_mask = obj_motion_data.get("mask_o_vz_abs", None)
    obj_x_motion_abs_mask = obj_motion_data.get("mask_o_vx_abs", None)
    obj_z_motion_rel_mask = obj_motion_data.get("mask_o_vz_rel", None)
    obj_x_motion_rel_mask = obj_motion_data.get("mask_o_vx_rel", None)
    obj_ranks  = obj_motion_data.get("obj_ranks",  {})
    obj_bboxes = obj_motion_data.get("obj_bboxes", {})
    
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "obj_speed_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def is_absent(s):
        return float(s[0]) == -1.0 and float(s[1]) == -1.0

    num_frames = len(frames_data)

    # ── unpack and align ego signals + masks to num_frames ──────────────────
    ego_vx_full = list(ego_data["ego_x_speeds"])
    ego_vz_full = list(ego_data["ego_z_speeds"])
    ego_stopped_full = list(ego_data.get("stopped_mask", [False] * num_frames))
    ego_turn_full    = list(ego_data.get("turn_mask",    [False] * num_frames))
    for _sig in (ego_vx_full, ego_vz_full, ego_stopped_full, ego_turn_full):
        if len(_sig) < num_frames:
            _sig += [_sig[-1]] * (num_frames - len(_sig))
    ego_vx_full      = [float(v) for v in ego_vx_full[:num_frames]]
    ego_vz_full      = [float(v) for v in ego_vz_full[:num_frames]]
    ego_stopped_full = [bool(v)  for v in ego_stopped_full[:num_frames]]
    ego_turn_full    = [bool(v)  for v in ego_turn_full[:num_frames]]

    # ── precompute per-object bbox/label lookup ──────────────────────────────
    obj_frame_info = {}
    for fi, frame_data in enumerate(frames_data):
        obj_ids_fi = frame_data.get("obj_ids", [])
        bboxes_fi  = frame_data.get("bboxes", [])
        labels_fi  = frame_data.get("labels", [])
        for k, oid in enumerate(obj_ids_fi):
            if oid not in obj_frame_info:
                obj_frame_info[oid] = {}
            bbox  = bboxes_fi[k].tolist() if hasattr(bboxes_fi[k], "tolist") else list(bboxes_fi[k])
            label = labels_fi[k] if k < len(labels_fi) else str(oid)
            obj_frame_info[oid][fi] = (bbox, label)

    # ── canvas dimensions ────────────────────────────────────────────────────
    first_img  = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    # ── per object ───────────────────────────────────────────────────────────
    for obj_id, speed_track in obj_speeds.items():
        speed_track = np.array(speed_track, dtype=np.float32)
        if speed_track.ndim == 1:
            speed_track = speed_track.reshape(1, 2)
        if len(speed_track) < num_frames:
            pad = np.full((num_frames - len(speed_track), 2), -1.0, dtype=np.float32)
            speed_track = np.vstack([speed_track, pad])
        speed_track = speed_track[:num_frames]

        # ── trim trailing absent frames ──────────────────────────────────────
        last_present = -1
        for _fi in range(len(speed_track) - 1, -1, -1):
            if not (float(speed_track[_fi][0]) == -1.0 and float(speed_track[_fi][1]) == -1.0):
                last_present = _fi
                break
        if last_present == -1:
            continue
        active_frames = last_present + 1
        speed_track   = speed_track[:active_frames]

        # z-motion track
        z_track = None
        if obj_z_motion_rel_mask is not None:
            z_track = list(obj_z_motion_rel_mask.get(obj_id, [-1] * num_frames))
            if len(z_track) < num_frames:
                z_track += [-1] * (num_frames - len(z_track))
            z_track = z_track[:active_frames]

        # x-motion track (relative)
        x_track = None
        if obj_x_motion_rel_mask is not None:
            x_track = list(obj_x_motion_rel_mask.get(obj_id, [-1] * num_frames))
            if len(x_track) < num_frames:
                x_track += [-1] * (num_frames - len(x_track))
            x_track = x_track[:active_frames]

        # z-motion track (absolute)
        z_abs_track = None
        if obj_z_motion_abs_mask is not None:
            z_abs_track = list(obj_z_motion_abs_mask.get(obj_id, [-1] * num_frames))
            if len(z_abs_track) < num_frames:
                z_abs_track += [-1] * (num_frames - len(z_abs_track))
            z_abs_track = z_abs_track[:active_frames]

        # x-motion track (absolute)
        x_abs_track = None
        if obj_x_motion_abs_mask is not None:
            x_abs_track = list(obj_x_motion_abs_mask.get(obj_id, [-1] * num_frames))
            if len(x_abs_track) < num_frames:
                x_abs_track += [-1] * (num_frames - len(x_abs_track))
            x_abs_track = x_abs_track[:active_frames]

        # ── build segments for obj vz chart (Approaching / Moving Away) ──────
        vz_segments:   list = []
        vz_seg_colors: list = []
        vz_seg_labels: list = []
        if z_track is not None:
            _i = 0
            while _i < len(z_track):
                _cat = z_track[_i]
                if _cat in (1, 2):
                    _j = _i
                    while _j < len(z_track) and z_track[_j] == _cat:
                        _j += 1
                    vz_segments.append((_i, _j))
                    vz_seg_colors.append('steelblue' if _cat == 1 else 'tomato')
                    vz_seg_labels.append('Approaching' if _cat == 1 else 'Moving Away')
                    _i = _j
                else:
                    _i += 1

        # ── build segments for obj vx chart (Left / Right) ───────────────────
        vx_segments:   list = []
        vx_seg_colors: list = []
        vx_seg_labels: list = []
        if x_track is not None:
            _i = 0
            while _i < len(x_track):
                _cat = x_track[_i]
                if _cat in (1, 2):
                    _j = _i
                    while _j < len(x_track) and x_track[_j] == _cat:
                        _j += 1
                    vx_segments.append((_i, _j))
                    vx_seg_colors.append('cornflowerblue' if _cat == 1 else 'salmon')
                    vx_seg_labels.append('Left' if _cat == 1 else 'Right')
                    _i = _j
                else:
                    _i += 1

        # ── object speed signals (NaN where absent) ───────────────────────────
        vz_signal = [float('nan') if is_absent(s) else float(s[1]) for s in speed_track]
        vx_signal = [float('nan') if is_absent(s) else float(s[0]) for s in speed_track]

        # ── ego signals and masks trimmed to active_frames ──────────────────
        ego_vz_signal     = ego_vz_full[:active_frames]
        ego_vx_signal     = ego_vx_full[:active_frames]
        ego_stopped_signal = ego_stopped_full[:active_frames]
        ego_turn_signal    = ego_turn_full[:active_frames]

        # build Stopped segments for ego vz chart
        ego_vz_segs:   list = []
        ego_vz_colors: list = []
        ego_vz_slabels: list = []
        _i = 0
        while _i < len(ego_stopped_signal):
            if ego_stopped_signal[_i]:
                _j = _i
                while _j < len(ego_stopped_signal) and ego_stopped_signal[_j]:
                    _j += 1
                ego_vz_segs.append((_i, _j))
                ego_vz_colors.append('orange')
                ego_vz_slabels.append('Stopped')
                _i = _j
            else:
                _i += 1

        # build Turning segments for ego vx chart
        ego_vx_segs:   list = []
        ego_vx_colors: list = []
        ego_vx_slabels: list = []
        _i = 0
        while _i < len(ego_turn_signal):
            if ego_turn_signal[_i]:
                _j = _i
                while _j < len(ego_turn_signal) and ego_turn_signal[_j]:
                    _j += 1
                ego_vx_segs.append((_i, _j))
                ego_vx_colors.append('mediumpurple')
                ego_vx_slabels.append('Turning')
                _i = _j
            else:
                _i += 1

        info_map   = obj_frame_info.get(obj_id, {})
        obj_label  = next((v[1] for v in info_map.values()), str(obj_id))
        safe_label = obj_label.replace(' ', '_').replace('/', '_')

        # ── probe chart cell dimensions once ─────────────────────────────────
        # Each cell is half the frame width; use chart_width/2 so matplotlib
        # proportions stay reasonable.
        cell_w = frame_w // 2
        sample_cell = utils.create_timeline_line_chart_img(
            vz_signal, 0, vz_segments, vz_seg_colors,
            f"{obj_label} Obj Vz",
            width=chart_width / 2, height=chart_height,
            segment_labels=vz_seg_labels,
        )
        cell_h  = int(sample_cell.shape[0] * cell_w / sample_cell.shape[1])
        total_h = frame_h + 2 * cell_h

        out_path = output_dir / f"{video_id}_obj_{obj_id}_{safe_label}_speed.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, frame_w / 1000)
        thickness  = max(2, int(font_scale * 2))
        pad        = int(10 * font_scale)

        def _draw_frame_axes(img):
            axis_len = max(36, int(min(frame_w, frame_h) * 0.08))
            origin_x = pad * 2
            origin_y = pad * 2
            label_scale = max(0.45, font_scale * 0.65)
            label_thickness = max(1, thickness - 1)
            axis_thickness = max(2, thickness)
            x_color = (255, 220, 80)
            z_color = (80, 220, 255)

            box_pad = max(6, pad // 2)
            box_x2 = origin_x + axis_len + int(24 * label_scale) + box_pad
            box_y2 = origin_y + axis_len + int(18 * label_scale) + box_pad
            axes_ov = img.copy()
            cv2.rectangle(
                axes_ov,
                (max(0, origin_x - box_pad), max(0, origin_y - box_pad)),
                (min(frame_w, box_x2), min(frame_h, box_y2)),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.addWeighted(axes_ov, 0.35, img, 0.65, 0, img)

            cv2.arrowedLine(
                img,
                (origin_x, origin_y),
                (origin_x + axis_len, origin_y),
                x_color,
                axis_thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.arrowedLine(
                img,
                (origin_x, origin_y),
                (origin_x, origin_y + axis_len),
                z_color,
                axis_thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.putText(
                img,
                "x",
                (origin_x + axis_len + box_pad // 2, origin_y + int(5 * label_scale)),
                font,
                label_scale,
                x_color,
                label_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "z",
                (origin_x - int(5 * label_scale), origin_y + axis_len + int(14 * label_scale)),
                font,
                label_scale,
                z_color,
                label_thickness,
                cv2.LINE_AA,
            )

        for fi, frame_data in enumerate(frames_data[:active_frames]):
            frame_img = utils.load_frame(frame_data["frame"])
            if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
                frame_img = cv2.resize(frame_img, (frame_w, frame_h))

            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
            _draw_frame_axes(frame_bgr)
            s = speed_track[fi]

            z_cat = z_track[fi] if z_track is not None else (0 if not is_absent(s) else -1)
            x_cat = x_track[fi] if x_track is not None else (0 if not is_absent(s) else -1)
            z_text, bbox_color = _Z_LABEL.get(z_cat, ("Unknown", (128, 128, 128)))
            x_text, x_color    = _X_LABEL.get(x_cat, ("Unknown", (128, 128, 128)))

            # absolute motion categories
            z_abs_cat = z_abs_track[fi] if z_abs_track is not None else (0 if not is_absent(s) else -1)
            x_abs_cat = x_abs_track[fi] if x_abs_track is not None else (0 if not is_absent(s) else -1)
            z_abs_text, _ = _Z_LABEL.get(z_abs_cat, ("Unknown", (128, 128, 128)))
            x_abs_text, _ = _X_LABEL.get(x_abs_cat, ("Unknown", (128, 128, 128)))

            obj_lbl_font_scale = font_scale * 1.4
            obj_lbl_thickness  = max(2, int(obj_lbl_font_scale * 2))

            # ── compute target bbox area for distance comparison ──────────────
            target_area = 0.0
            if fi in info_map:
                _tb = info_map[fi][0]
                _tx1, _ty1, _tx2, _ty2 = int(_tb[0]), int(_tb[1]), int(_tb[2]), int(_tb[3])
                target_area = max(0, _tx2 - _tx1) * max(0, _ty2 - _ty1)

            # ── draw all other objects with relative-distance color ───────────
            # Uses obj_bboxes {obj_id: {fi: bbox}} built by percept2obj_dist_rank_and_bboxes.
            # Larger bbox area → closer than target → red (255,0,0); smaller → farther → gray.
            for _oid, _fi_bbox_map in obj_bboxes.items():
                if _oid == obj_id:
                    continue   # target drawn separately below
                _bb = _fi_bbox_map.get(fi)
                if _bb is None:
                    continue
                _bb = _bb.tolist() if hasattr(_bb, "tolist") else list(_bb)
                _ox1, _oy1, _ox2, _oy2 = int(_bb[0]), int(_bb[1]), int(_bb[2]), int(_bb[3])
                _area = max(0, _ox2 - _ox1) * max(0, _oy2 - _oy1)
                # closer (larger area) → red; farther (smaller area) → yellow
                _other_color = (0, 0, 255) if _area > target_area else (0, 220, 220)
                _other_thickness = thickness * 2
                cv2.rectangle(frame_bgr, (_ox1, _oy1), (_ox2, _oy2), _other_color, _other_thickness)
                # rank label above bbox
                _o_rank_track = obj_ranks.get(_oid, [])
                _o_rank_val   = _o_rank_track[fi] if fi < len(_o_rank_track) else -1
                _o_rank_str   = f"Rank #{_o_rank_val}" if _o_rank_val >= 1 else ""
                # look up label from obj_frame_info
                _olbl = obj_frame_info.get(_oid, {}).get(fi, (None, ""))[1]
                # combine rank + label into two rows (rank on top, label below)
                _label_y = max(_oy1 - 4, 10)
                if _olbl:
                    cv2.putText(frame_bgr, _olbl, (_ox1, _label_y),
                                font, font_scale * 0.7, _other_color, _other_thickness, cv2.LINE_AA)
                    _label_y = max(_label_y - int(font_scale * 0.7 * 20) - 2, 10)
                if _o_rank_str:
                    cv2.putText(frame_bgr, _o_rank_str, (_ox1, _label_y),
                                font, font_scale * 0.7, _other_color, _other_thickness, cv2.LINE_AA)

            # ── draw target object with green bbox + abs/rel labels above ─────
            if fi in info_map:
                bbox, lbl = info_map[fi]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                _target_border = (0, 200, 0)   # green for target
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _target_border, 2)

                # rank of this object among all tracked objects (1 = closest)
                _rank_track = obj_ranks.get(obj_id, [])
                _rank_val   = _rank_track[fi] if fi < len(_rank_track) else -1
                _rank_str   = f"Rank #{_rank_val}" if _rank_val >= 0 else "Rank: ?"

                # three label rows above bbox:
                #   row 1 (highest): rank
                #   row 2: abs z + abs x
                #   row 3 (closest): rel z + rel x
                rank_row = _rank_str
                abs_row  = f"Abs: {z_abs_text} | {x_abs_text}"
                rel_row  = f"Rel: {z_text} | {x_text}"
                (_nw, _nh), _nbl = cv2.getTextSize(rank_row, font, obj_lbl_font_scale, obj_lbl_thickness)
                (_aw, _ah), _abl = cv2.getTextSize(abs_row,  font, obj_lbl_font_scale, obj_lbl_thickness)
                (_rw, _rh), _rbl = cv2.getTextSize(rel_row,  font, obj_lbl_font_scale, obj_lbl_thickness)
                row_gap = 4
                total_label_h = (_nh + _nbl) + row_gap + (_ah + _abl) + row_gap + (_rh + _rbl)
                rel_y  = max(y1 - 4, total_label_h + 4)
                abs_y  = rel_y  - (_rh + _rbl + row_gap)
                rank_y = abs_y  - (_ah + _abl + row_gap)

                # semi-transparent background behind all three rows
                bg_x1 = max(0, x1)
                bg_x2 = min(frame_w, x1 + max(_nw, _aw, _rw) + 8)
                bg_y1 = max(0, rank_y - _nh - 2)
                bg_y2 = min(frame_h, rel_y + _rbl + 2)
                lbl_ov = frame_bgr.copy()
                cv2.rectangle(lbl_ov, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(lbl_ov, 0.4, frame_bgr, 0.6, 0, frame_bgr)

                cv2.putText(frame_bgr, rank_row, (x1, rank_y),
                            font, obj_lbl_font_scale, (0, 200, 0), obj_lbl_thickness, cv2.LINE_AA)
                cv2.putText(frame_bgr, abs_row,  (x1, abs_y),
                            font, obj_lbl_font_scale, (200, 200, 200), obj_lbl_thickness, cv2.LINE_AA)
                cv2.putText(frame_bgr, rel_row,  (x1, rel_y),
                            font, obj_lbl_font_scale, bbox_color, obj_lbl_thickness, cv2.LINE_AA)
            else:
                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (40, 40, 40), cv2.FILLED)
                cv2.addWeighted(overlay, 0.25, frame_bgr, 0.75, 0, frame_bgr)

            # ── ego state label — bottom-left ─────────────────────────────────
            ego_z_text  = "Stopped" if ego_stopped_signal[fi] else "Moving"
            ego_z_color = (255, 165, 0)   if ego_stopped_signal[fi] else (200, 200, 200)
            ego_x_text  = "Turning" if ego_turn_signal[fi] else "Straight"
            ego_x_color = (180, 100, 220) if ego_turn_signal[fi] else (200, 200, 200)

            e_lines = [("Ego", (255, 255, 255)), (ego_z_text, ego_z_color), (ego_x_text, ego_x_color)]
            e_sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t, _ in e_lines]
            ebox_w  = max(sz[0][0] for sz in e_sizes) + 3 * pad
            ebox_h  = sum(sz[0][1] + sz[1] for sz in e_sizes) + (len(e_lines) + 1) * pad
            eby0    = frame_h - ebox_h - pad   # top of box anchored to bottom
            ego_ov  = frame_bgr.copy()
            cv2.rectangle(ego_ov, (pad, eby0), (pad + ebox_w, eby0 + ebox_h), (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(ego_ov, 0.45, frame_bgr, 0.55, 0, frame_bgr)
            ey = eby0
            for (etxt, ecol), ((_etw, _eth), _ebl) in zip(e_lines, e_sizes):
                ey += _eth + pad
                cv2.putText(frame_bgr, etxt, (2 * pad, ey), font, font_scale, ecol, thickness, cv2.LINE_AA)
                ey += _ebl

            frame_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            fi_str = f"frame {fi}/{active_frames - 1}"
            if not is_absent(s):
                vz_speed_str = f"  vz={float(s[1]):.3f} m/s  rel:{z_text}  abs:{z_abs_text}"
                vx_speed_str = f"  vx={float(s[0]):.3f} m/s  rel:{x_text}  abs:{x_abs_text}"
            else:
                vz_speed_str = vx_speed_str = "  absent"

            # ── 2×2 chart grid ────────────────────────────────────────────────
            # top-left : obj vx   (Left/Right shading)
            obj_vx_cell = utils.create_timeline_line_chart_img(
                vx_signal, fi, vx_segments, vx_seg_colors,
                f"{obj_label} Obj vx{vx_speed_str}",
                width=chart_width / 2, height=chart_height,
                segment_labels=vx_seg_labels,
            )
            # bottom-left: ego vx  (Turning shading)
            ego_vx_cell = utils.create_timeline_line_chart_img(
                ego_vx_signal, fi, ego_vx_segs, ego_vx_colors,
                f"{fi_str}  ego vx={ego_vx_signal[fi]:.3f} m/s [Ego vx]",
                width=chart_width / 2, height=chart_height,
                segment_labels=ego_vx_slabels,
            )
            # top-right : obj vz   (Approaching/Moving Away shading)
            obj_vz_cell = utils.create_timeline_line_chart_img(
                vz_signal, fi, vz_segments, vz_seg_colors,
                f"{obj_label} Obj vz{vz_speed_str}",
                width=chart_width / 2, height=chart_height,
                segment_labels=vz_seg_labels,
            )
            # bottom-right: ego vz  (Stopped shading)
            ego_vz_cell = utils.create_timeline_line_chart_img(
                ego_vz_signal, fi, ego_vz_segs, ego_vz_colors,
                f"{fi_str}  ego vz={ego_vz_signal[fi]:.3f} m/s [Ego vz]",
                width=chart_width / 2, height=chart_height,
                segment_labels=ego_vz_slabels,
            )

            # resize all cells to (cell_w, cell_h) and tile into 2×2
            obj_vx_cell = cv2.resize(obj_vx_cell, (cell_w, cell_h))
            ego_vx_cell = cv2.resize(ego_vx_cell, (cell_w, cell_h))
            obj_vz_cell = cv2.resize(obj_vz_cell, (cell_w, cell_h))
            ego_vz_cell = cv2.resize(ego_vz_cell, (cell_w, cell_h))

            left_col  = np.vstack([obj_vx_cell, ego_vx_cell])   # obj vx top, ego vx bottom
            right_col = np.vstack([obj_vz_cell, ego_vz_cell])   # obj vz top, ego vz bottom
            chart_panel = np.hstack([left_col, right_col])      # 2×2 grid, full frame_w wide
            # guard against 1-pixel rounding difference (odd frame_w)
            if chart_panel.shape[1] != frame_w:
                chart_panel = cv2.resize(chart_panel, (frame_w, 2 * cell_h))

            composite = np.vstack([frame_img, chart_panel])
            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Saved object speed visualization ({active_frames} frames) → {out_path}")


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

def visualize_ego_speed(frames_data, ego_x_speeds, ego_z_speeds, video_id,
                        stopped_mask=None, turning_mask=None,
                        fps=5, stop_threshold=0.01,
                        chart_width=10, chart_height=3,
                        output_dir=None):
    """
    Produce an MP4 where every frame shows:
      - top panel    : original video frame
      - bottom panel : ego-speed line chart with a red dot at the current frame,
                       and orange-shaded regions where |speed| < stop_threshold.

    Output: *output_dir*/<video_id>_ego_speed.mp4

    Parameters
    ----------
    frames_data : list of dict
        As returned by raw2frame_data — one dict per frame with a "frame" key.
    ego_x_speeds : list of float
        Per-frame x-axis speed values aligned to frames_data (length == len(frames_data)).
    ego_z_speeds : list of float
        Per-frame z-axis speed values aligned to frames_data (length == len(frames_data)).
    video_id : str
    fps : int
        Output video frame rate.
    stop_threshold : float
        Frames whose |speed| is below this value are shaded as "stopped".
    chart_width : float
        Matplotlib figure width (inches) for the speed chart.
    chart_height : float
        Matplotlib figure height (inches) for the speed chart.
    output_dir : Path-like or None
        Destination directory; defaults to pipeline_output/ego_speed_vis/.
    """
    import pathlib

    num_frames = len(frames_data)
    
    ego_speeds = [np.sqrt(x**2 + z**2) for x, z in zip(ego_x_speeds, ego_z_speeds)]
    # guard against length mismatch
    ego_speeds = list(ego_speeds)
    if len(ego_speeds) < num_frames:
        ego_speeds += [ego_speeds[-1]] * (num_frames - len(ego_speeds))
    ego_speeds = ego_speeds[:num_frames]

    # ── determine canvas dimensions ───────────────────────────────────────────
    first_img = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    sample_chart = utils.create_timeline_line_chart_img(
        ego_speeds, 0, [], [], "Ego Speed (m/s)",
        width=chart_width, height=chart_height,
        stopped_mask=stopped_mask
    )
    chart_h = int(sample_chart.shape[0] * frame_w / sample_chart.shape[1])
    total_h = frame_h + chart_h

    # ── open VideoWriter ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "ego_speed_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_ego_speed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

    # ── write every frame ─────────────────────────────────────────────────────
    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))

        chart_img = utils.create_timeline_line_chart_img(
            ego_speeds, fi, [], [],
            f"Ego Speed (m/s) — frame {fi}/{num_frames - 1}  |  speed = {ego_speeds[fi]:.4f} m/s",
            width=chart_width, height=chart_height,
            stopped_mask=stopped_mask
        )
        chart_img = cv2.resize(chart_img, (frame_w, chart_h))

        composite = np.vstack([frame_img, chart_img])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved ego speed visualization ({num_frames} frames) → {out_path}")
    return out_path

def visualize_mask(frames_data, ego_speeds, ego_rotations, stopped_mask, turning_mask, video_id,
                   fps=5, chart_width=10, chart_height=3, output_dir=None):
    """
    Produce an MP4 with three stacked panels per frame:
      - top    : original video frame
      - middle : ego-speed line chart; orange shading where mask == 1 (stopped)
      - bottom : ego yaw-rate line chart; orange shading where mask == 2 (turning)

    mask values: 0 = driving, 1 = stopped, 2 = turning  (as returned by merge_segs)

    Output: *output_dir*/<video_id>_mask_vis.mp4
    """
    import pathlib

    num_frames = len(frames_data)

    # ── align signal lengths to num_frames ────────────────────────────────────
    def _align(seq):
        seq = list(seq)
        if len(seq) < num_frames:
            seq += [seq[-1]] * (num_frames - len(seq))
        return seq[:num_frames]

    ego_speeds    = _align(ego_speeds)
    ego_rotations = _align(ego_rotations)
    stopped_mask  = _align(stopped_mask)
    turning_mask  = _align(turning_mask)

    # ── determine canvas dimensions ───────────────────────────────────────────
    first_img = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    sample_speed_chart = utils.create_timeline_line_chart_img(
        ego_speeds, 0, [], [], "Ego Speed (m/s)",
        width=chart_width, height=chart_height,
        stopped_mask=stopped_mask,
    )
    speed_chart_h = int(sample_speed_chart.shape[0] * frame_w / sample_speed_chart.shape[1])

    sample_rot_chart = utils.create_timeline_line_chart_img(
        ego_rotations, 0, [], [], "Ego Yaw-Rate (rad/s)",
        width=chart_width, height=chart_height,
        stopped_mask=turning_mask,
    )
    rot_chart_h = int(sample_rot_chart.shape[0] * frame_w / sample_rot_chart.shape[1])

    total_h = frame_h + speed_chart_h + rot_chart_h

    # ── open VideoWriter ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "mask_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_mask_vis.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

    # state label config: mask value → (text, BGR colour)
    _STATE_LABEL = {
        0: ("Normal",  (100, 200, 100)),   # green
        1: ("Stopped", (0,   100, 255)),   # orange-red
        2: ("Turning", (255, 100,   0)),   # blue
    }

    # ── write every frame ─────────────────────────────────────────────────────
    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))

        # overlay state label top-left (work in BGR for cv2)
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        if stopped_mask[fi]:
            mask_val = 1
        elif turning_mask[fi]:
            mask_val = 2
        else:            
            mask_val = 0
        state_text, label_color = _STATE_LABEL.get(mask_val, ("Unknown", (128, 128, 128)))
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, frame_w / 800)
        thickness  = max(2, int(font_scale * 2))
        pad        = int(10 * font_scale)
        (tw, th), baseline = cv2.getTextSize(state_text, font, font_scale, thickness)
        # semi-transparent dark background behind text
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay,
                      (pad, pad),
                      (pad + tw + pad, pad + th + baseline + pad),
                      (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)
        cv2.putText(frame_bgr, state_text,
                    (pad * 2, pad + th),
                    font, font_scale, label_color, thickness, cv2.LINE_AA)
        frame_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        speed_chart = utils.create_timeline_line_chart_img(
            ego_speeds, fi, [], [],
            f"Ego Speed (m/s) — frame {fi}/{num_frames - 1}  |  speed = {ego_speeds[fi]:.4f} m/s",
            width=chart_width, height=chart_height,
            stopped_mask=stopped_mask,
        )
        speed_chart = cv2.resize(speed_chart, (frame_w, speed_chart_h))

        rot_chart = utils.create_timeline_line_chart_img(
            ego_rotations, fi, [], [],
            f"Ego Yaw-Rate (rad/s) — frame {fi}/{num_frames - 1}  |  yaw = {ego_rotations[fi]:.4f} rad/s",
            width=chart_width, height=chart_height,
            stopped_mask=turning_mask,
        )
        rot_chart = cv2.resize(rot_chart, (frame_w, rot_chart_h))

        composite = np.vstack([frame_img, speed_chart, rot_chart])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved mask visualization ({num_frames} frames) → {out_path}")
    return out_path

def visualize_ego_rotation(frames_data, ego_rotations, video_id,
                           turn_mask=None,
                           fps=5, chart_width=10, chart_height=3,
                           output_dir=None):
    """
    Produce an MP4 where every frame shows:
      - top panel    : original video frame
      - bottom panel : ego yaw-rate line chart with a red dot at the current
                       frame and orange-shaded regions for turning frames.

    Output: *output_dir*/<video_id>_ego_rotation.mp4

    Parameters
    ----------
    frames_data : list of dict
        As returned by raw2frame_data — one dict per frame with a "frame" key.
    ego_rotations : list of float
        Per-frame smoothed yaw-rate values aligned to frames_data.
    video_id : str
    turn_mask : list of bool or None
        Per-frame turn flags as returned by percept2turn_mask.  Turning frames
        are shaded orange in the chart.  Pass None to disable.
    fps : int
        Output video frame rate.
    chart_width : float
        Matplotlib figure width (inches) for the rotation chart.
    chart_height : float
        Matplotlib figure height (inches) for the rotation chart.
    output_dir : Path-like or None
        Destination directory; defaults to pipeline_output/ego_rotation_vis/.
    """
    import pathlib

    num_frames = len(frames_data)

    # guard against length mismatch
    ego_rotations = list(ego_rotations)
    if len(ego_rotations) < num_frames:
        ego_rotations += [ego_rotations[-1]] * (num_frames - len(ego_rotations))
    ego_rotations = ego_rotations[:num_frames]

    # ── determine canvas dimensions ───────────────────────────────────────────
    first_img = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    sample_chart = utils.create_timeline_line_chart_img(
        ego_rotations, 0, [], [], "Ego Yaw-Rate (rad/s)",
        width=chart_width, height=chart_height,
        stopped_mask=turn_mask
    )
    chart_h = int(sample_chart.shape[0] * frame_w / sample_chart.shape[1])
    total_h = frame_h + chart_h

    # ── open VideoWriter ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "ego_rotation_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_ego_rotation.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

    # ── write every frame ─────────────────────────────────────────────────────
    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))

        chart_img = utils.create_timeline_line_chart_img(
            ego_rotations, fi, [], [],
            f"Ego Yaw-Rate (rad/s) — frame {fi}/{num_frames - 1}  |  yaw = {ego_rotations[fi]:.4f} rad/s",
            width=chart_width, height=chart_height,
            stopped_mask=turn_mask
        )
        chart_img = cv2.resize(chart_img, (frame_w, chart_h))

        composite = np.vstack([frame_img, chart_img])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved ego rotation visualization ({num_frames} frames) → {out_path}")
    return out_path

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
    return motion_cfg, vis_cfg
    
    
    
def run_single_video(video_id, cfg=None):
    out_path = get_out_path(video_id)
    pipeline_data = load_pipeline_data(out_path, video_id)
    motion_cfg, vis_cfg = _get_cfgs(out_path, cfg)
    
    # ----------- Estimate ego motion masks (stopped, turning) and visualize them -----------
    print(f"Estimating Ego mask for video: {video_id}")
    video_data = _load_or_compute_video_data(video_id, out_path, pipeline_data)
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
    
