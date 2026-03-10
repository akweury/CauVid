
import numpy as np 
import matplotlib.pyplot as plt
import os 
import imageio
import cv2

import config 
from src.exp_driving_videos import exp_driving_utils as utils
from src.exp_driving_videos import knowledge 

def load_matrix(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        matrix = pickle.load(f)
    return matrix




def matrix2primitives(matrix):
    # Placeholder for the actual implementation of converting matrices to primitives
    # This function should take the input matrix and convert it into a list of primitives
    # For example, it could extract object positions, velocities, and other relevant information
    primitives = {}
    for obj_id, obj_data in matrix.items():
        label = obj_data.get('label', 'unknown')  # Get the label, default to 'unknown' if not present
        positions = np.array(obj_data.get('position', None))  # Get the position, default to None if not present
        frame_indices = obj_data.get('frame_indices', [])  # Get the frame indices, default to an empty list if not present
        primitives[obj_id] = {
            "moving_direction": knowledge.position2direction(positions),
            "velocity": knowledge.position2velocity(positions),
            "label": label,
            "bboxes": obj_data.get('bboxes', []),  # Get the bounding boxes, default to an empty list if not present
            "frames": obj_data.get('frames', []),  # Get the frames, default to an empty list if not present
            "frame_indices": frame_indices  # Include frame indices in the primitives
        }
    return primitives





def estimate_ego_velocity(frame_non_ego_velocities, frame_non_ego_labels):
    # Placeholder for the actual implementation of estimating ego velocity based on the velocities of other objects in the frame
    # This function should take the velocities of the other objects and estimate the ego velocity based on their relative velocities
    non_ego_velocities = np.array(frame_non_ego_velocities)
    
    mask_pedestrian = np.array(frame_non_ego_labels) == "pedestrian"
    mask_vehicle = np.array(frame_non_ego_labels) == "vehicle"
    
    if np.any(mask_pedestrian):
        non_ego_velocities = non_ego_velocities[mask_pedestrian]
    elif np.any(mask_vehicle):
        non_ego_velocities = non_ego_velocities[mask_vehicle]
    else:
        non_ego_velocities = np.array(frame_non_ego_velocities)  # If no specific labels, use all velocities
    
    vx = non_ego_velocities[:, 0] if non_ego_velocities.size > 0 else np.array([])
    vz = non_ego_velocities[:, 2] if non_ego_velocities.size > 0 else np.array([])    
    ego_vx = -np.median(vx) if vx.size > 0 else 0
    ego_vz = -np.median(vz) if vz.size > 0 else 0

    return np.array([ego_vx, 0, ego_vz])  # Assuming the y-axis velocity is zero for simplicity

    
    
    

def estimate_ego_moving_direction(frame_ego_velocity, threshold_steps= 100):
    # estimate directions along two axes: left/right and forward/backward
    
    # return a Nx2 vector, N is the threshold steps, 
    # the first column is the left/right direction, the second column is the forward/backward direction
    # calculate the moving direction under N different thresholds, 
    # and return the table of moving direction under different thresholds, 
    # which can be used to analyze the sensitivity of the moving direction estimation to the threshold selection.
    
    thresholds = np.linspace(0, np.max(np.abs(frame_ego_velocity)), threshold_steps)
    moving_directions = []
    for threshold in thresholds:
        moving_direction = np.zeros(2)  # [left/right, forward/backward]
        if frame_ego_velocity[0] > threshold:
            moving_direction[0] = 1  # right
        elif frame_ego_velocity[0] < -threshold:
            moving_direction[0] = -1  # left
        else:
            moving_direction[0] = 0  # stationary in left/right direction
        
        if frame_ego_velocity[2] > threshold:
            moving_direction[1] = 1  # forward
        elif frame_ego_velocity[2] < -threshold:
            moving_direction[1] = -1  # backward
        else:
            moving_direction[1] = 0  # stationary in forward/backward direction
        
        moving_directions.append(moving_direction)
    
    return np.array(moving_directions)


def estimate_ego_primitives(obj_primitives):
    """ 
    This function takes the object primitives and converts them into frame primitives.
    
    First, the input is a dictionary of object primitives, it is not organized by frame, but by object.
    We need to recognize the input and convert it into a 2d matrix with frame index as the rows and object id as the columns. 
    Each cell in the matrix contains the primitive information for that object in that frame.
    
    Then, estimate the ego primitives based on the other objects in the frame.
    For example,  estimate the ego velocity and moving direction based on the relative position and velocity of the other objects in the frame.
    
    The output is a 2d matrix with frame index as the rows and object id as the columns, 
    each cell contains the primitive information for that object in that frame, 
    and also the estimated ego primitives based on the other objects in the frame.
    The ego primitives is placed in the first column (object id 0) of the matrix, 
    and the other objects are placed in the following columns (object id 1, 2, ...).    
    """
    def smooth_ego_velocity(ego_velocities, window_size=5):
        # Smooth the ego velocities using a moving average filter
        smoothed_velocities = np.copy(ego_velocities)
        for i in range(len(ego_velocities)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ego_velocities), i + window_size // 2 + 1)
            smoothed_velocities[i] = np.mean(ego_velocities[start_idx:end_idx], axis=0)
        return smoothed_velocities
    
    
    total_obj_num = len(obj_primitives)
    frame_num = max([max(primitives["frame_indices"]) for primitives in obj_primitives.values()]) + 1  # Get the total number of frames based on the maximum frame index    
    all_frame_filenames = [name for primitives in obj_primitives.values() for name in primitives["frames"]]
    all_frame_filenames = list(set(all_frame_filenames))  # Get unique frame filenames
    all_frame_filenames.sort()  # Sort the frame filenames to ensure correct order
    object_frame_primitives = np.empty((frame_num, total_obj_num), dtype=object)  # Create an empty array to hold frame primitives
    
    # reassign id to make it from 0 to total_obj_num-1
    obj_id_mapping = {obj_id: idx for idx, obj_id in enumerate(obj_primitives.keys())}
    for obj_id, primitives in obj_primitives.items():
        new_obj_id = obj_id_mapping[obj_id]
        for frame_index, moving_direction, velocity, bbox, frame in zip(
            primitives["frame_indices"], 
            primitives["moving_direction"], 
            primitives["velocity"], 
            primitives.get("bboxes", [None] * len(primitives["frame_indices"])),
            primitives.get("frames", [None] * len(primitives["frame_indices"]))
        ):
            object_frame_primitives[frame_index, new_obj_id] = {
                "moving_direction": moving_direction,
                "velocity": velocity,
                "label": primitives["label"],
                "bbox": bbox,
                "frame": frame
            }
        
    
    # Estimate the ego velocity based on the other objects velocity in the same frame
    ego_frame_velocities = []
    for frame_index in range(frame_num):
        frame_objects_data = object_frame_primitives[frame_index]
        frame_non_ego_velocities = [obj_prim["velocity"] for obj_prim in frame_objects_data if obj_prim is not None]
        obj_labels = [obj_prim["label"] for obj_prim in frame_objects_data if obj_prim is not None]
        ego_velocity = estimate_ego_velocity(frame_non_ego_velocities, obj_labels)
        ego_frame_velocities.append(ego_velocity)
    
    # smooth the ego velocities using a moving average filter
    ego_frame_velocities_smoothed = smooth_ego_velocity(np.array(ego_frame_velocities))
    # Estimate the ego moving direction based on the smoothed ego velocity
    ego_frame_moving_directions = [estimate_ego_moving_direction(ego_velocity) 
                                   for ego_velocity in ego_frame_velocities_smoothed]
    
    ego_frame_primitives = np.empty(frame_num, dtype=object)
    for frame_index in range(frame_num):
        ego_frame_primitives[frame_index] = {
            "moving_direction": ego_frame_moving_directions[frame_index],
            "velocity": ego_frame_velocities_smoothed[frame_index],
            "label": "ego_vehicle",
            "frame": all_frame_filenames[frame_index]
        }
    return ego_frame_primitives
    
    
    
    
def visualize_ego_primitives(ego_primitives, output_path):
    os.makedirs(output_path, exist_ok=True)
    # This function should take the list of ego primitives and create a visualization, such as a plot or animation
    # For example, we can create a plot of the ego velocity and moving
    
    def create_direction_threshold_analysis_img(velocity, moving_direction, frame_index, title_suffix=""):
        
        # the moving direction is a Nx2 vector, the first column is the left/right direction, 
        # the second column is the forward/backward direction. N is the number of thresholds. 
        
        # Create two subfigures side by side:
        # - Left subplot: threshold forward/backward vs value of forward/backward
        # - Right subplot: threshold left/right vs value of left/right
        
        max_threshold = np.max(np.abs(velocity))
        min_threshold = 0 
        thresholds = np.linspace(min_threshold, max_threshold, len(moving_direction))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left subplot: Forward/Backward
        ax1.step(thresholds, moving_direction[:, 1], color="blue", label="Forward/Backward", linewidth=2, where='post')
        ax1.set_xlabel('Threshold Forward/Backward', fontsize=12)
        ax1.set_ylabel('Direction', fontsize=12)
        ax1.set_xlim(min_threshold, max_threshold)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(['Backward', 'Stationary', 'Forward'])
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Forward/Backward Direction', fontsize=12)
        ax1.legend(loc='upper right')
        
        # Right subplot: Left/Right
        ax2.step(thresholds, moving_direction[:, 0], color="orange", label="Left/Right", linewidth=2, where='post')
        ax2.set_xlabel('Threshold Left/Right', fontsize=12)
        ax2.set_ylabel('Direction', fontsize=12)
        ax2.set_xlim(min_threshold, max_threshold)
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Left', 'Stationary', 'Right'])
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Left/Right Direction', fontsize=12)
        ax2.legend(loc='upper right')
        
        # Add overall title
        fig.suptitle(f'Ego Moving Direction Analysis {title_suffix}', fontsize=14)
        plt.tight_layout()
        
        # convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Convert RGBA to RGB
        plt.close(fig)
        return img
    
    visual_frames = []
    for frame_index, primitive in enumerate(ego_primitives):
        velocity = primitive["velocity"]
        moving_direction = primitive["moving_direction"]
        primitive_img = create_direction_threshold_analysis_img(velocity, moving_direction, frame_index)
        frame_img = utils.load_frame(primitive["frame"], None, "ego", "ego_vehicle")  # Load the frame for the ego vehicle
        
        # Resize primitive image to match frame height
        if primitive_img.shape[0] != frame_img.shape[0]:
            aspect_ratio = primitive_img.shape[1] / primitive_img.shape[0]
            new_width = int(frame_img.shape[0] * aspect_ratio)
            primitive_img = cv2.resize(primitive_img, (new_width, frame_img.shape[0]))
        combined_img = np.hstack((frame_img, primitive_img))
        visual_frames.append(combined_img)

    # save the visual frames as a gif, with 2 fps
    gif_path = output_path / f"ego_primitives_{len(ego_primitives)}frames.gif"
    imageio.mimsave(gif_path, visual_frames, fps=2)
    print(f"Saved ego primitive gif: {gif_path}")
    
    
def visualize_others_primitives(primitives, output_path):
    
    # Actual implementation of visualizing primitives
    os.makedirs(output_path, exist_ok=True)

    def create_primitive_img_current(velocity, moving_direction, frame_index, obj_id, label, frame_height, title_suffix=""):
        # Create a visualization for the current primitive (e.g., velocity and moving direction)
        # show the current moving direction as an arrow, and the velocity as the length of the arrow
        img_width = frame_height
        img = np.ones((frame_height, img_width, 3), dtype=np.uint8) * 255  # White background   
        center_z = frame_height // 2
        center_x = img_width // 2
        # Scale the velocity for visualization (you may need to adjust the scaling factor)
        scaling_factor = 1000
        arrow_length = int(scaling_factor * np.linalg.norm(velocity))
        # Determine the end point of the arrow based on the moving direction
        end_x = center_x + int(arrow_length * moving_direction)
        # point the arrow up for forward velocity, down for backward velocity, and no vertical movement for stationary
        end_z = center_z - int(arrow_length * (velocity[2] / np.linalg.norm(velocity))) if np.linalg.norm(velocity) > 0 else center_z
        # Draw the arrow on the image
        cv2.arrowedLine(img, (center_x, center_z), (end_x, end_z), (255, 0, 0), 5)
        # Add text for velocity and moving direction
        cv2.putText(img, f"Velocity: {np.linalg.norm(velocity):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        direction_text = "Right" if moving_direction > 0 else "Left" if moving_direction < 0 else "Stationary"
        cv2.putText(img, f"Direction: {direction_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Object {obj_id} ({label}) {title_suffix}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
        
        
    
    # This function should take the list of primitives and create a visualization, such as a plot or animation
    # two pannels: left side showing the primitives per frame, right side showing the video frames with bounding boxes and labels
    for obj_id, obj_primitives in primitives.items():
        label = obj_primitives["label"]
        frames = obj_primitives["frames"]
        velocities = obj_primitives["velocity"]
        moving_directions = obj_primitives["moving_direction"]
        bboxes = obj_primitives.get("bboxes", [None] * len(frames))  # Get bounding boxes if available, otherwise use None
        # each bbox is one frame, so we can visualize the trajectory and the frame with bbox side by side
        visual_frames = []
        for i, (velocity, moving_direction, frame, bbox) in enumerate(zip(velocities, moving_directions, frames, bboxes)):
            frame_img = utils.load_frame(frame, bbox, obj_id, label)  # Load the frame and draw the bounding box
            frame_height = frame_img.shape[0]
            
            primitive_img_current = create_primitive_img_current(
                velocity, moving_direction,
                i, obj_id, label, frame_height, 
                title_suffix=""
            )
            # Resize primitive image to match frame height
            if primitive_img_current.shape[0] != frame_img.shape[0]:
                aspect_ratio = primitive_img_current.shape[1] / primitive_img_current.shape[0]
                new_width = int(frame_img.shape[0] * aspect_ratio)
                primitive_img_current = cv2.resize(primitive_img_current, (new_width, frame_img.shape[0]))
            # Combine trajectory and frame side by side
            combined_img = np.hstack((primitive_img_current, 
                                      frame_img))
            visual_frames.append(combined_img)

        # save the visual frames as a gif, with 2 fps
        safe_label = label.replace(' ', '_').replace('/', '_')
        gif_path = output_path / f"obj_{obj_id}_{safe_label}_{len(moving_directions)}frames.gif"
        imageio.mimsave(gif_path, visual_frames, fps=2)
        print(f"Saved primitive gif for Object {obj_id} ({label}): {gif_path}")
    
    
    
if __name__ == "__main__":    
    # load processed matrices
    obj_matrix = load_matrix(config.get_output_path("pipeline_output") / "smoothed_object_matrices.pkl")    
    # convert matrices to primitives
    obj_primitives = matrix2primitives(obj_matrix)
    
    # frame primitives
    ego_primitives = estimate_ego_primitives(obj_primitives)
    
    # visualize other object primitives
    # visualize_others_primitives(obj_primitives, config.get_output_path("driving_primitive_visualization"))
    
    # visual ego primitives
    visualize_ego_primitives(ego_primitives, config.get_output_path("driving_ego_primitive_visualization"))
    
    print("finished processing primitives.")