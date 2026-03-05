# created on 02-03-2026
# Perception to Matrix
# Input: Video frames and corresponding bounding boxes, labels and dpeth maps
# Output: A matrix representation of the scene for each frame, including object positions, labels, and depth information
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.exp_driving_videos import exp_driving_utils as utils
import torch 

import config 

def raw2frame_data(input_data):
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
    frame_folder = dataset_path / "frames" / config.driving_demo_video_id
    
    for i in tqdm(range(len(depth_maps))):
        
        frame_i_indices = torch.tensor(frame_indices) == i
        video_i_indices = [video_name == config.driving_demo_video_id for video_name in video_names]
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
    
        # debug: save each frame with bboxes drawn on it
        debug_visual_frame(frame_full_path, frame_i_bboxes, frame_i_labels)
    
    return frames_data
def debug_visual_frame(frame_path, bboxes, labels):
    """
    Visualize a single frame with bounding boxes and labels for debugging purposes.

    Parameters:
    frame_path (str): The path to the frame image.
    bboxes (list): A list of bounding boxes for the objects in the frame.
    labels (list): A list of labels corresponding to the bounding boxes.

    Returns:
    None: Displays the visualized frame with bounding boxes and labels.
    """
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # save the visualized frame for debugging
    debug_output_folder = config.OUTPUT_PATHS["temp"]/"debug_frames"
    debug_output_folder.mkdir(parents=True, exist_ok=True)
    
    debug_output_path = debug_output_folder / f"debug_{frame_path.name}"
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Debug Frame: {frame_path.name}", fontsize=14)
    plt.savefig(debug_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved debug visual frame: {debug_output_path}")
    
    
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
    
    

def raw2matrix(raw_frame_input, min_frame_number=5):
    frames_data = raw2frame_data(raw_frame_input)
    obj_matrices = {}
    for frame_data in frames_data:
        # Process the frame data to create a matrix representation
        # use frame image, depth map, bounding boxes, labels to create a matrix representation of the scene
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
                    "bboxes": [bbox]
                }
            else:
                obj_matrices[obj_id]["position"].append(position)
                obj_matrices[obj_id]["bboxes"].append(bbox)
                obj_matrices[obj_id]["frames"].append(frame)
    # filter out objects that appear in less than min_frame_number frames
    obj_matrices = {obj_id: data for obj_id, data in obj_matrices.items() if len(data["position"]) >= min_frame_number}
    
    return obj_matrices

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
        
        smoothed_position = smooth_positions(position)  # Smooth the position across frames
        smoothed_matrices[obj_id] = {
            "label": label,
            "position": smoothed_position
        }
    return smoothed_matrices

def trajectory_visual(matrices, output_path):
    """
    Visualize trajectories of objects in the scene (top-down view).
    Creates individual figures for each object and saves them.
    
    Parameters:
    matrices (dict): A dictionary containing object matrices with positions.
    output_path (Path): Path to the folder where trajectory visualizations will be saved.
    """
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create individual figure for each object
    for obj_id, data in matrices.items():
        label = data["label"]
        position = torch.stack(data["position"])
        x = position[:, 0].numpy()
        z = position[:, 2].numpy()
        
        # Create a new figure for this object
        plt.figure(figsize=(10, 10))
        plt.plot(x, z, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.xlabel("X (meters)", fontsize=12)
        plt.ylabel("Z (meters)", fontsize=12)
        plt.title(f"Top-down Trajectory: Object {obj_id} ({label})", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add start and end markers
        plt.scatter(x[0], z[0], color='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(x[-1], z[-1], color='red', s=100, marker='X', label='End', zorder=5)
        plt.legend()
        
        # Save the figure
        safe_label = label.replace(' ', '_').replace('/', '_')
        filename = f"trajectory_obj{obj_id}_{safe_label}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory for Object {obj_id} ({label}): {save_path}")
        
        # Close the figure to free memory
        plt.close()

def trajectory_with_frames_visual(matrices, input_data, output_path):
    os.makedirs(output_path, exist_ok=True)
    # for each trajectory, show the smoothed trajectory on the left, highlight the current position, 
    # and the image on the right,bounding box the the target object, saved as a gif file
    
    def create_trajectory_figure_with_current(position, current_idx, obj_id, label, target_height):
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
        ax.set_title(f"Trajectory: Object {obj_id} ({label})\nFrame {current_idx+1}/{len(position)}", 
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
        position = torch.stack(data["position"])
        frames = data["frames"]
        bboxes = data["bboxes"]
        
        # each bbox is one frame, so we can visualize the trajectory and the frame with bbox side by side
        visual_frames = []
        for i, (bbox, frame) in enumerate(zip(bboxes, frames)):
            # Load and process the frame
            img = cv2.imread(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_height = img.shape[0]
            
            # Draw bounding box on frame
            frame_copy = img.copy()
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(frame_copy, f"Object {obj_id} ({label})", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Create trajectory plot with current position highlighted
            trajectory_img_current = create_trajectory_figure_with_current(
                position, i, obj_id, label, frame_height
            )
            
            # Combine trajectory and frame side by side
            combined_img = np.hstack((trajectory_img_current, frame_copy))
            visual_frames.append(combined_img)
        
        # save the visual frames as a gif, with 2 fps
        safe_label = label.replace(' ', '_').replace('/', '_')
        gif_path = output_path / f"obj_{obj_id}_{safe_label}_{len(bboxes)}frames.gif"
        imageio.mimsave(gif_path, visual_frames, fps=2)
        print(f"Saved trajectory gif for Object {obj_id} ({label}): {gif_path}")

    
     

if __name__ == "__main__":
    # Example usage
    
    
    input_data = utils.load_driving_mini_inputs()
    matrices = raw2matrix(input_data)    
    
    # trajectory_visual(matrices, config.get_output_path("driving_trajectory_visualization"))
    print("Generated object matrices for the scene:")
    print("Number of objects:", len(matrices))
    
    smooth_matrices = smooth_matrices(matrices)
    # trajectory_visual(smooth_matrices, config.get_output_path("driving_trajectory_visualization_smoothed"))
    
    
    # for each trajectory, show the smoothed trajectory on the left, highlight the current position, 
    # and the image on the right,bounding box the the target object, saved as a gif file
    trajectory_with_frames_visual(matrices, input_data, config.get_output_path("driving_trajectory_visualization_with_frames")) 
    
    
    