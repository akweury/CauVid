import os
from copy import deepcopy
from pathlib import Path

import numpy as np 
try:
    import torch
except ModuleNotFoundError:
    torch = None
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ModuleNotFoundError:
    plt = None
    mpatches = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, *args, **kwargs: iterable

import config


def merge_cfg(base, override):
    cfg = deepcopy(base)
    if not override:
        return cfg
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = merge_cfg(cfg[key], value)
        else:
            cfg[key] = value
    return cfg


def parse_yaml_scalar(value):
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [parse_yaml_scalar(item.strip()) for item in items.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml_cfg(cfg_path):
    root = {}
    stack = [(-1, root)]
    with open(cfg_path, "r") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            key, sep, value = raw_line.strip().partition(":")
            if not sep:
                continue
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            value = value.strip()
            if value == "":
                child = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = parse_yaml_scalar(value)
    return root


def load_pattern_cfg_file(cfg_path):
    try:
        from omegaconf import OmegaConf

        loaded = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(loaded, resolve=True) or {}
    except ModuleNotFoundError:
        try:
            import yaml

            with open(cfg_path, "r") as f:
                return yaml.safe_load(f) or {}
        except ModuleNotFoundError:
            return load_simple_yaml_cfg(cfg_path)


def get_pattern_cfg(cfg=None):
    base_cfg = load_pattern_cfg_file(config.get_config_path("pattern_mining"))
    if isinstance(cfg, (str, Path)):
        cfg = load_pattern_cfg_file(cfg)
    return merge_cfg(base_cfg, cfg or {})

def load_video_frames(frame_path):
    """
    Load video frames from a specified path.

    Parameters:
    frame_path (str): The path to the video frames.

    Returns:
    list: A list of loaded video frame image names.
    """
    import os
    frames = []
    for filename in sorted(os.listdir(frame_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frames.append(filename)
    return frames


def load_labels(label_csv_file):
    """ load bounding boxes and labels from a csv file 
    columns: 
    name (videoName-frameindex_7_digits.jpg), 
    videoName, 
    frameindex,
    id,
    category,
    attributes.crowd,
    attributes.occluded,
    attributes.truncated,
    box2d.x1,
    box2d.x2,
    box2d.y1,
    box2d.y2,
    haveVideo
    
    """
    import pandas as pd

    df = pd.read_csv(label_csv_file)
    bboxes = []
    labels = []
    obj_ids = []
    video_names = []
    frame_indices = []
    for _, row in df.iterrows():
        bbox = [int(row['box2d.x1']), int(row['box2d.y1']), int(row['box2d.x2']), int(row['box2d.y2'])]
        label = row['category']
        obj_id = row['id']
        bboxes.append(bbox)
        labels.append(label)
        obj_ids.append(obj_id)
        video_names.append(row['videoName'])
        frame_indices.append(row['frameIndex'])
    return bboxes, labels, obj_ids, video_names, frame_indices
        
        
def load_depth_npz(path, device="cpu", target_size=None):
    """
    Load depth map from npz file and optionally resize to target size
    
    Args:
        path: Path to the .npz file
        device: PyTorch device ('cpu' or 'cuda')
        target_size: Optional tuple (height, width) to resize depth map to match input image
    
    Returns:
        torch.Tensor: Depth map of shape (H, W)
    """
    data = np.load(path)
    
    # Usually only one key
    key = data.files[0]
    depth = data[key]
    
    # Remove singleton dimensions if present
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    
    depth = torch.from_numpy(depth).float().to(device)
    
    # Resize if target size is provided and doesn't match current size
    if target_size is not None:
        current_size = depth.shape  # (H, W)
        if current_size != target_size:
            import torch.nn.functional as F
            # Add batch and channel dimensions for interpolation
            depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            # Resize using bilinear interpolation
            depth = F.interpolate(depth, size=target_size, mode='bilinear', align_corners=False)
            # Remove batch and channel dimensions
            depth = depth.squeeze(0).squeeze(0)  # (H, W)
    
    return depth

def load_depth_maps(depth_map_folder):
    """
    Load depth maps from a specified path.

    Parameters:
    depth_map_path (str): The path to the depth map data.

    Returns:
    list: A list of depth map file names.
    """
    import os 
    depth_maps = []
    for filename in sorted(os.listdir(depth_map_folder)):
        if filename.endswith('.npz'):
            depth_maps.append(os.path.join(depth_map_folder, filename))
    return depth_maps

def load_perception_inputs(frame_path, depth_map_path, label_file_path):
    """
    Load perception inputs including video frames, bounding boxes, labels, and depth maps.

    Parameters:
    frame_path (str): The path to the video frames.
    bbox_path (str): The path to the bounding box data.
    label_path (str): The path to the label data.
    depth_map_path (str): The path to the depth map data.

    Returns:
    tuple: A tuple containing lists of frames, bounding boxes, labels, object IDs, video names, frame indices, and depth maps.
    """
    frames = load_video_frames(frame_path)
    bboxes, labels, obj_ids, video_names, frame_indices = load_labels(label_file_path)
    depth_map_file_names = load_depth_maps(depth_map_path)  # Assuming depth maps are stored in a similar format as bounding boxes

    input_data = {
        'frames': frames,
        'bboxes': bboxes,
        'labels': labels,
        'video_names': video_names,
        'frame_indices': frame_indices,
        'depth_map_file_names': depth_map_file_names,
        'obj_ids': obj_ids
    }
    return input_data


def load_driving_mini_inputs(video_id=None):
    dataset_path = config.DATASET_PATHS['driving_mini'] 
    if video_id is None:
        frame_path = dataset_path / "frames"/config.driving_demo_video_id  
        depth_map_path = dataset_path / "depth_maps"/config.driving_demo_video_id
    else:
        frame_path = dataset_path / "frames"/video_id
        depth_map_path = dataset_path / "depth_maps"/video_id
    label_file_path =dataset_path/"labels.csv"
    input_data = load_perception_inputs(frame_path, depth_map_path, label_file_path)
    return input_data


def load_frame(frame, bbox=None, obj_id=None, label=None):
    # Load and process the frame
    img = cv2.imread(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if bbox is not None:
        # Draw bounding box on frame
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img, f"Object {obj_id} ({label})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return img 


def load_matrix(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        matrix = pickle.load(f)
    return matrix

if __name__ == "__main__":
    dataset_path = config.DATASET_PATHS['driving_mini'] 
    frame_path = dataset_path / "frames"/config.driving_demo_video_id  
    depth_map_path = dataset_path / "depth_maps"/config.driving_demo_video_id
    label_file_path =dataset_path/"labels.csv"

    input_data = load_perception_inputs(frame_path, depth_map_path, label_file_path)

    print("Loaded frames:", len(input_data['frames']))
    print("Loaded depth maps:", len(input_data['depth_map_file_names']))
    print("Loaded bounding boxes:", len(input_data['bboxes']))
    print("Loaded labels:", len(input_data['labels']))
    print("Loaded object IDs:", len(input_data['obj_ids']))
    
    
    
def create_line_chart_img(x, y, title, xlabel, ylabel):
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(x, y, marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    w, h = int(renderer.width), int(renderer.height)
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = img[:, :, :3]
    plt.close(fig)
    return img

def combine_images_vertically(image_list):
    """
    Combine a list of images vertically into a single image.

    Parameters:
    image_list (list): A list of images (numpy arrays) to combine.

    Returns:
    numpy.ndarray: The combined image.
    """
    # Ensure all images have the same width
    widths = [img.shape[1] for img in image_list]
    max_width = max(widths)
    
    # resize images to have the same width
    resized_images = []
    for img in image_list:
        if img.shape[1] != max_width:
            scale_factor = max_width / img.shape[1]
            new_height = int(img.shape[0] * scale_factor)
            resized_img = cv2.resize(img, (max_width, new_height))
            resized_images.append(resized_img)
        else:
            resized_images.append(img)

    
    # Concatenate images vertically
    combined_image = np.vstack(resized_images)
    
    return combined_image



def combine_images_horizontally(image_list):
    """
    Combine a list of images horizontally into a single image.

    Parameters:
    image_list (list): A list of images (numpy arrays) to combine.

    Returns:
    numpy.ndarray: The combined image.
    """
    # Ensure all images have the same height
    heights = [img.shape[0] for img in image_list]
    max_height = max(heights)
    
    # Pad images to have the same height
    padded_images = []
    for img in image_list:
        if img.shape[0] < max_height:
            padding = max_height - img.shape[0]
            padded_img = np.pad(img, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=255)
            padded_images.append(padded_img)
        else:
            padded_images.append(img)
    
    # Concatenate images horizontally
    combined_image = np.hstack(padded_images)
    
    return combined_image

def create_segment_feature_img(segments, frame_index, colors, title, width=6, height=5):
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    for si, seg in enumerate(segments):
        s, e, feat, label = seg['start'], seg['end'], seg['features'], seg['label']
        ax.axvspan(s, e, color=colors[si], alpha=0.3)
        ax.text((s+e)/2, 0.5, f"{si}\n{label}", ha='center', va='center', fontsize=10)
        
    # highlight current frame
    ax.axvline(frame_index, color='red', linestyle='--', label='Current Frame')
    ax.set_xlabel("Frame Index")
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    w, h = int(renderer.width), int(renderer.height)
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = img[:, :, :3]
    plt.close(fig)
    return img
    
def create_timeline_line_chart_img(signal, frame_index, segments, colors, title, width=10, height=5, stopped_mask=None, segment_labels=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    x = list(range(len(signal)))
    y = signal
    ax.plot(x, y, marker='o')
    ax.scatter(frame_index, signal[frame_index], color='red', s=100, label='Current Frame')
    # shade named segments, deduplicating legend entries per label
    seen_seg_labels = set()
    has_seg_legend = False
    for si, seg in enumerate(segments):
        s, e = seg
        lbl = '_nolegend_'
        if segment_labels is not None and si < len(segment_labels):
            raw_lbl = segment_labels[si]
            if raw_lbl not in seen_seg_labels:
                lbl = raw_lbl
                seen_seg_labels.add(raw_lbl)
                has_seg_legend = True
        ax.axvspan(s, e, color=colors[si], alpha=0.3, label=lbl)
    if has_seg_legend:
        ax.legend(fontsize=8)
    if stopped_mask is not None:
        # shade every consecutive run of True entries orange
        in_stop = False
        stop_start = None
        first_stop_span = True
        for xi, is_stopped in enumerate(stopped_mask):
            if is_stopped and not in_stop:
                stop_start = xi
                in_stop = True
            elif not is_stopped and in_stop:
                label_str = 'Stopped' if first_stop_span else '_nolegend_'
                ax.axvspan(stop_start, xi, color='orange', alpha=0.25, label=label_str)
                in_stop = False
                first_stop_span = False
        if in_stop:
            label_str = 'Stopped' if first_stop_span else '_nolegend_'
            ax.axvspan(stop_start, x[-1], color='orange', alpha=0.25, label=label_str)
        ax.legend(fontsize=8)
    ax.set_xlabel("Frame Index")
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    w, h = int(renderer.width), int(renderer.height)
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = img[:, :, :3]
    plt.close(fig)
    return img


def create_state_bar_chart_img(rotations_list, state_threshold=0.1):        
    """
    Create a square visualization showing ego rotation states across time.
    The bar spans from top to bottom and extends horizontally across frames.

    Colors:
    green = straight
    red   = left turn
    blue  = right turn
    """
    states = []
    for r in rotations_list:
        if r > state_threshold:
            states.append("right")
        elif r < -state_threshold:
            states.append("left")
        else:
            states.append("straight")

    colors = {
        "straight": (0.2, 0.8, 0.2),  # green
        "left": (0.9, 0.2, 0.2),      # red
        "right": (0.2, 0.4, 0.9)      # blue
    }

    # Build color strip
    color_strip = np.array([colors[s] for s in states])
    color_strip = color_strip[np.newaxis, :, :]  # shape (1, N, 3)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Stretch vertically to make a fat bar
    ax.imshow(color_strip, aspect='auto', extent=[0, len(states), 0, 1])

    ax.set_xlim(0, len(states))
    ax.set_ylim(0, 1)

    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_ylabel("Rotation State", fontsize=12)
    ax.set_title("Ego Rotation State History", fontsize=12)

    ax.set_yticks([])

    # Legend
    legend_handles = [
        mpatches.Patch(color=colors["straight"], label="Straight"),
        mpatches.Patch(color=colors["left"], label="Left Turn"),
        mpatches.Patch(color=colors["right"], label="Right Turn")
    ]

    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    w, h = int(renderer.width), int(renderer.height)
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = img[:, :, :3]

    plt.close(fig)
    return img

def save_pkl_file(filename, data):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(data, f)    
        
        
        
def write_video(frames, video_path, fps=2):
    import imageio
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
def visual_pca(X,hue, figure_path):
    from sklearn.decomposition import PCA
    import seaborn as sns
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=hue,
        palette="tab10",
        alpha=0.7,
    )
    plt.title("PCA Colored by Cluster")
    
    png_file_name = config.get_output_path('driving_seg_feat_vis') / f"drive_mini_span_records_PCA.png"
    plt.savefig(png_file_name)
    plt.close()
    
    
def create_bg_mask(frame, bboxes, labels, obj_ids):
    """
    create a binary mask for the background by excluding the bounding boxes of detected objects.
    """
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # start with a white mask
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0), thickness=-1)  # fill object area with black
    return mask
    
def compute_optical_flow(frame1, frame2, device=None):
    """
    Compute dense optical flow between two RGB frames using torchvision's
    pretrained RAFT-Small model.

    Parameters
    ----------
    frame1 : np.ndarray
        First frame, shape (H, W, 3), dtype uint8, RGB.
    frame2 : np.ndarray
        Second frame, same shape/dtype as frame1.
    device : str or torch.device or None
        Target device. Defaults to 'cuda' if available, else 'cpu'.

    Returns
    -------
    flow : np.ndarray
        Shape (H, W, 2), float32.
        flow[y, x, 0] = horizontal displacement (dx),
        flow[y, x, 1] = vertical displacement   (dy).
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    import torch.nn.functional as F

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device).eval()
    transforms = weights.transforms()

    def to_tensor(img):
        # (H, W, 3) uint8 -> (1, 3, H, W) uint8
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    t1, t2 = transforms(to_tensor(frame1), to_tensor(frame2))
    t1, t2 = t1.to(device), t2.to(device)

    # RAFT requires spatial dims divisible by 8
    H, W = t1.shape[-2], t1.shape[-1]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        t1 = F.pad(t1, (0, pad_w, 0, pad_h))
        t2 = F.pad(t2, (0, pad_w, 0, pad_h))

    with torch.no_grad():
        # returns list of iteratively refined predictions; last is finest
        flow_predictions = model(t1, t2)
        flow_tensor = flow_predictions[-1]  # (1, 2, H_pad, W_pad)

    # crop padding back to original size, convert to (H, W, 2) numpy float32
    flow_tensor = flow_tensor[:, :, :H, :W]
    return flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()



def estimate_ego_speed(flow, bg_mask, depth_map):
    """
    Estimate ego speed based on the optical flow of the background between frames.
    
    Parameters:
    flow: np.ndarray of shape (H, W, 2) - optical flow vectors (dx, dy) for each pixel
    bg_mask: np.ndarray of shape (H, W) - binary mask where 255 indicates background pixels
    depth_map: np.ndarray of shape (H, W) - depth values for each pixel
    
    Returns:
    float: Estimated ego speed in meters per second
    """
    # Extract background flow vectors
    bg_flow = flow[bg_mask == 255]  # shape (N_bg, 2)
    bg_depth = depth_map[bg_mask == 255]  # shape (N_bg,)
    
    # Filter out invalid depth values (e.g., zero or negative)
    valid_mask = bg_depth > 0
    bg_flow = bg_flow[valid_mask]
    bg_depth = bg_depth[valid_mask].numpy()
    
    if len(bg_flow) == 0:
        return 0.0  # No valid background pixels, cannot estimate speed
    
    # Compute the magnitude of the flow vectors
    flow_magnitudes = np.linalg.norm(bg_flow, axis=1)  # shape (N_valid_bg,)
    
    # Convert pixel flow to real-world speed using depth information
    focal_length_pixels = 1000.0  # Example focal length in pixels (needs calibration)
    
    # Speed estimation formula: speed = (flow_magnitude * depth) / focal_length
    speeds = (flow_magnitudes * bg_depth) / focal_length_pixels
    
    # Average speed across all valid background pixels
    ego_speed = np.median(speeds)
    return ego_speed

def estimate_ego_motion(flow, bg_mask, depth_map,
                        focal_length: float = 1000.0,
                        angle_tol_deg: float = 35.0,
                        min_flow_px: float = 0.5,
                        return_yaw: bool = False):
    """
    Estimate ego translation (vx, vz) from background optical flow with robust
    rotation filtering via vector-angle consensus.

    Algorithm
    ---------
    1. Extract valid background pixels (bg_mask == 255, depth > 0, flow magnitude
       above ``min_flow_px``).
    2. Angle consensus: compute the circular-median of all flow directions.
       Keep inliers whose angle deviates by less than ``angle_tol_deg``.
       - Pure rotation adds a *uniform* horizontal bias to every flow vector,
         which shifts the consensus direction but preserves angular coherence;
         independently-moving foreground objects scatter in random directions
         and are rejected by this step without any spatial mask.
    3. Depth-weighted translation on the inlier set:
         vz = median( |flow_i| * Z_i )   — forward / total speed (≥ 0)
         vx = median(  dx_i   * Z_i )   — lateral component (signed)
    4. Yaw residual (only when ``return_yaw=True``):
       The translational lateral prediction at pixel i is  vx / Z_i.
       The horizontal residual  dx_i - vx/Z_i  is depth-independent for pure
       yaw (≈ -f * ω_y), so its median gives the yaw rate estimate:
         yaw_rate = -median(dx_residual) / focal_length

    Parameters
    ----------
    flow         : np.ndarray (H, W, 2)   optical flow (dx, dy)
    bg_mask      : np.ndarray (H, W)      255 = background pixels
    depth_map    : np.ndarray or Tensor (H, W)
    focal_length : float   camera focal length in pixels (for yaw conversion)
    angle_tol_deg: float   half-angle of the inlier cone around consensus direction
    min_flow_px  : float   discard pixels with flow magnitude below this value
    return_yaw   : bool    if True, also return yaw rate as third element

    Returns
    -------
    (vx, vz)             if return_yaw is False  (default, backward-compatible)
    (vx, vz, yaw_rate)   if return_yaw is True
    """
    if hasattr(depth_map, 'numpy'):
        depth_map = depth_map.numpy()
    depth_map = np.asarray(depth_map, dtype=np.float32)

    # ── 1. valid background pixels ────────────────────────────────────────────
    valid = (bg_mask == 255) & (depth_map > 0)
    if valid.sum() == 0:
        return (0.0, 0.0, 0.0) if return_yaw else (0.0, 0.0)

    dx = flow[valid, 0].astype(np.float32)
    dy = flow[valid, 1].astype(np.float32)
    Z  = depth_map[valid]

    # discard near-zero flow (static pixels, numerical noise)
    mag = np.hypot(dx, dy)
    moving = mag > min_flow_px
    if moving.sum() < 5:
        return (0.0, 0.0, 0.0) if return_yaw else (0.0, 0.0)
    dx, dy, Z, mag = dx[moving], dy[moving], Z[moving], mag[moving]

    # ── 2. angle consensus → inlier set ──────────────────────────────────────
    angles = np.arctan2(dy, dx)

    # circular median: more robust than circular mean under heavy-tailed noise
    theta_consensus = np.arctan2(np.median(np.sin(angles)),
                                 np.median(np.cos(angles)))

    angle_diff = angles - theta_consensus
    # wrap to [-π, π]
    angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi

    tol = np.deg2rad(angle_tol_deg)
    inliers = np.abs(angle_diff) < tol

    # fallback: if consensus is too narrow, accept all
    if inliers.sum() < max(5, int(0.05 * len(dx))):
        inliers = np.ones(len(dx), dtype=bool)

    dx_in = dx[inliers]
    dy_in = dy[inliers]
    Z_in  = Z[inliers]

    # ── 3. depth-weighted translation ────────────────────────────────────────
    vz = float(np.median(np.hypot(dx_in, dy_in) * Z_in))   # forward speed ≥ 0
    vx = float(np.median(dx_in * Z_in))                     # lateral (signed)

    if not return_yaw:
        return vx, vz

    # ── 4. yaw from horizontal residual ──────────────────────────────────────
    # translational prediction of horizontal flow at each inlier: vx / Z_i
    dx_trans_pred = vx / Z_in
    dx_residual   = dx_in - dx_trans_pred
    yaw_rate = float(-np.median(dx_residual) / focal_length)

    return vx, vz, yaw_rate

def estimate_ego_rotation(flow, bg_mask, depth_map):
    """
    Estimate ego rotation based on the optical flow of the background between frames.
    
    Parameters:
    flow: np.ndarray of shape (H, W, 2) - optical flow vectors (dx, dy) for each pixel
    bg_mask: np.ndarray of shape (H, W) - binary mask where 255 indicates background pixels
    depth_map: np.ndarray of shape (H, W) - depth values for each pixel
    
    Returns:
    float: Estimated ego rotation in radians per second (positive for right turn, negative for left turn)
    """
    # Extract background flow vectors
    bg_flow = flow[bg_mask == 255]  # shape (N_bg, 2)
    bg_depth = depth_map[bg_mask == 255]  # shape (N_bg,)
    
    # Filter out invalid depth values (e.g., zero or negative)
    valid_mask = bg_depth > 0
    bg_flow = bg_flow[valid_mask]
    bg_depth = bg_depth[valid_mask].numpy()
    
    if len(bg_flow) == 0:
        return 0.0  # No valid background pixels, cannot estimate rotation
    
    # Compute the horizontal component of the flow vectors
    horizontal_flow = bg_flow[:, 0]  # dx component
    
    # Convert pixel flow to real-world angular velocity using depth information
    focal_length_pixels = 1000.0  # Example focal length in pixels (needs calibration)
    
    # Rotation estimation formula: rotation = (horizontal_flow * depth) / focal_length
    rotations = (horizontal_flow * bg_depth) / focal_length_pixels
    
    # Average rotation across all valid background pixels
    ego_rotation = np.median(rotations)
    
    return ego_rotation


def raw2frame_data(input_data, video_id=None):
    """
    Convert perception data to a matrix representation.

    Parameters:
    input_data (dict): A dictionary containing perception data, including frames, bounding boxes, labels, and depth maps.

    Returns:
    list: A list of matrices representing the scene for each frame.
    """
    import config
    from pathlib import Path
    
    frames_data = []
    frames = input_data['frames']
    bboxes = input_data['bboxes']
    labels = input_data['labels']
    obj_ids = input_data['obj_ids']
    video_names = input_data['video_names']
    frame_indices = input_data['frame_indices']
    depth_maps = input_data['depth_map_file_names']
    
    # Get the frame path from config
    dataset_path = config.DATASET_PATHS['driving_mini']
    
    if video_id is None:
        video_id = config.driving_demo_video_id
    frame_folder = dataset_path / "frames" / video_id
    
    
    for i in tqdm(range(len(depth_maps))):
        
        frame_i_indices = torch.tensor(frame_indices) == i
        video_i_indices = [video_name == video_id for video_name in video_names]
        frame_mask = frame_i_indices & torch.tensor(video_i_indices)
        frame_i_bboxes = torch.tensor(bboxes)[frame_mask]
        frame_i_labels = [labels[j] for j in range(len(labels)) if frame_mask[j]]
        frame_i_obj_ids = [obj_ids[j] for j in range(len(obj_ids)) if frame_mask[j]]
        
        # Get full frame path
        frame_full_path = frame_folder / frames[i]
        
        frame_input = {
            "frame": str(frame_full_path),
            "depth_map": depth_maps[i],
            "bboxes": frame_i_bboxes,
            "labels": frame_i_labels,
            "obj_ids": frame_i_obj_ids
        }
        frames_data.append(frame_input)
    return frames_data

def extract_bg_mask(frames_data, video_id=None):
    
    bg_masks = []
    for frame_data in frames_data:
        frame = load_frame(frame_data["frame"])
        bboxes = frame_data["bboxes"]
        labels = frame_data["labels"]
        obj_ids = frame_data["obj_ids"]
        
        # Create a background mask for the frame
        bg_mask = create_bg_mask(frame, bboxes, labels, obj_ids)
        bg_masks.append(bg_mask)
    return bg_masks

def extract_flow(frames_data, video_id=None):
    flow_file = config.get_output_path("pipeline_output") / f"optical_flow_{video_id}.npy"
    if flow_file.exists():
        print(f"Loading cached optical flow: {flow_file}")
        flows = np.load(flow_file)
    else:
        flows = []
        for i in range(len(frames_data)-1):
            frame1 = load_frame(frames_data[i]["frame"])
            frame2 = load_frame(frames_data[i+1]["frame"])
            # raft optical flow
            flow = compute_optical_flow(frame1, frame2)
            flows.append(flow)
        np.save(flow_file, flows)
    return flows

def load_video_data(out_path, video_id):
    
    video_data_file = out_path / f"video_data.pkl"
    if video_data_file.exists():
        print(f"Loading cached video data: {video_data_file}")
        video_data = load_matrix(video_data_file)
        return video_data
    else:
        input_data  = load_driving_mini_inputs(video_id)
        frames_data = raw2frame_data(input_data, video_id)
        bg_mask     = extract_bg_mask(frames_data, video_id)
        depth_maps  = [fd["depth_map"] for fd in frames_data]
        flows       = extract_flow(frames_data, video_id)
        video_data = {
            'frames_data': frames_data,
            'bg_masks': bg_mask,
            'depth_maps': depth_maps,
            'flows': flows
        }
        save_pkl_file(video_data_file, video_data)
    return video_data
    
def estimate_obj_velo(flow, depth_map_file, bg_mask, bbox, ego_vx, ego_vz):
    x1, y1, x2, y2 = bbox
    depth_map = load_depth_npz(depth_map_file, target_size=bg_mask.shape).numpy()
    obj_region = np.zeros_like(bg_mask)
    obj_region[y1:y2, x1:x2] = 1

    obj_mask = obj_region & (~bg_mask) & (depth_map > 0)
    obj_mask = obj_mask.astype(bool)

    if obj_mask.sum() == 0:
        return 0.0, 0.0, False

    fx = flow[..., 0]
    fy = flow[..., 1]

    vx = depth_map * fx - ego_vx
    vz = depth_map * fy - ego_vz

    vx_vals = vx[obj_mask]
    vz_vals = vz[obj_mask]

    # robust filtering
    vx_vals = vx_vals[np.abs(vx_vals) < np.percentile(np.abs(vx_vals), 90)]
    vz_vals = vz_vals[np.abs(vz_vals) < np.percentile(np.abs(vz_vals), 90)]

    vx_obj = np.median(vx_vals)
    vz_obj = np.median(vz_vals)

    speed = np.sqrt(vx_obj**2 + vz_obj**2)
    variance = np.var(vx_vals) + np.var(vz_vals)

    moving = (speed > 0.1) or (variance > 0.05)

    return vx_obj, vz_obj, moving 
