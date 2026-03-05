import torch 
import numpy as np 

import config

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


def load_driving_mini_inputs():
    dataset_path = config.DATASET_PATHS['driving_mini'] 
    frame_path = dataset_path / "frames"/config.driving_demo_video_id  
    depth_map_path = dataset_path / "depth_maps"/config.driving_demo_video_id
    label_file_path =dataset_path/"labels.csv"
    input_data = load_perception_inputs(frame_path, depth_map_path, label_file_path)
    return input_data

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