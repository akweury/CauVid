import os

import torch 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    
def create_timeline_line_chart_img(signal, frame_index, segments, colors, title, width=10, height=5):
    
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    x = list(range(len(signal)))
    y = signal
    ax.plot(x, y, marker='o')
    ax.scatter(frame_index, signal[frame_index], color='red', s=100, label='Current Frame')
    for si, seg in enumerate(segments):
        s, e = seg['start'], seg['end']
        ax.axvspan(s, e, color=colors[si], alpha=0.3)
        # ax.text((s+e)/2, max(signal)*0.8, f"{label}", ha='center', va='center', fontsize=12)
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
    