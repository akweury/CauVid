
import numpy as np 
import matplotlib.pyplot as plt
import os 
import imageio
import cv2
from tqdm import tqdm

import config 
from src.exp_driving_videos import exp_driving_utils as utils
from src.exp_driving_videos import knowledge 

def load_matrix(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        matrix = pickle.load(f)
    return matrix




def matrix2obj_data(matrix):
    #
    objs_data = {}
    for obj_id, obj_data in matrix.items():
        label = obj_data.get('label', 'unknown')  # Get the label, default to 'unknown' if not present
        positions = np.array(obj_data.get('position', None))  # Get the position, default to None if not present
        frame_indices = obj_data.get('frame_indices', [])  # Get the frame indices, default to an empty list if not present
        objs_data[obj_id] = {
            "positions": positions,
            "moving_direction": knowledge.position2direction(positions),
            "velocity": knowledge.position2velocity(positions),
            "label": label,
            "bboxes": obj_data.get('bboxes', []),  # Get the bounding boxes, default to an empty list if not present
            "frames": obj_data.get('frames', []),  # Get the frames, default to an empty list if not present
            "frame_indices": frame_indices  # Include frame indices in the primitives
        }
    return objs_data





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
    if ego_vz<0:
        print(f"Warning: Estimated ego velocity in z direction is negative ({ego_vz:.2f}), which may indicate an issue with the estimation. Please check the input velocities and labels for accuracy.")
    return np.array([ego_vx, 0, ego_vz])  # Assuming the y-axis velocity is zero for simplicity


def estimate_ego_rotation(frame_non_ego_velocities, frame_non_ego_labels):
    """
    Estimate ego yaw rate (rotation around vertical axis).

    Convention used in this project:
        positive yaw  → right turn
        negative yaw  → left turn

    Idea:
    When the ego vehicle rotates right, static objects appear to move left
    in the camera frame. Therefore their observed lateral velocity vx is
    opposite to the ego rotation direction.

    So we approximate:

        omega ≈ - vx / (|vz| + eps)

    where
        vx : lateral velocity of objects
        vz : forward velocity magnitude (used for normalization)

    We use a robust median across objects to suppress outliers.
    """

    eps = 1e-3
    non_ego_velocities = np.array(frame_non_ego_velocities)

    if non_ego_velocities.size == 0:
        return 0.0

    vx = non_ego_velocities[:, 0]
    vz = non_ego_velocities[:, 2]

    # avoid unstable samples
    mask = np.abs(vz) > eps
    if not np.any(mask):
        return 0.0

    vx = vx[mask]
    vz = vz[mask]

    # corrected sign: negative because scene motion is opposite ego motion
    omega_candidates = -vx / (np.abs(vz) + eps)

    # remove extreme outliers
    omega_candidates = omega_candidates[np.abs(omega_candidates) < 0.5]

    if omega_candidates.size == 0:
        return 0.0

    # robust median
    ego_yaw_rate = np.median(omega_candidates)

    return float(ego_yaw_rate)

    
    
    

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
    def smooth_ego_velocity(ego_velocities, alpha=0.2):
        """
        Exponential moving average smoothing for ego velocity.
        alpha controls smoothing strength (smaller = smoother).
        """
        smoothed = np.zeros_like(ego_velocities)
        smoothed[0] = ego_velocities[0]
        for i in range(1, len(ego_velocities)):
            smoothed[i] = alpha * ego_velocities[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed
    
    
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
        for frame_index, moving_direction, velocity, position, bbox, frame in zip(
            primitives["frame_indices"],
            primitives["moving_direction"],
            primitives["velocity"],
            primitives["positions"],
            primitives.get("bboxes", [None] * len(primitives["frame_indices"])),
            primitives.get("frames", [None] * len(primitives["frame_indices"]))
        ):
            object_frame_primitives[frame_index, new_obj_id] = {
                "moving_direction": moving_direction,
                "velocity": velocity,
                "position": position,
                "label": primitives["label"],
                "bbox": bbox,
                "frame": frame
            }
        
    
    # Store ego rotation history
    ego_frame_rotations = []
    # Estimate the ego velocity and rotation using median-based strategy for each frame
    ego_frame_velocities = []
    for frame_index in range(frame_num):

        frame_objects_data = object_frame_primitives[frame_index]

        velocities = []
        for obj_prim in frame_objects_data:
            if obj_prim is None:
                continue
            vel = obj_prim.get("velocity", None)
            if vel is None:
                continue
            velocities.append(vel)

        velocities = np.array(velocities)

        if len(velocities) == 0:
            ego_velocity = np.array([0.0, 0.0, 0.0])
            ego_rotation = 0.0
        else:
            vx = velocities[:, 0]
            vz = velocities[:, 2]

            # median strategy: assume most objects are static
            ego_vx = -np.median(vx)
            ego_vz = -np.median(vz)

            ego_velocity = np.array([ego_vx, 0, ego_vz])

            # estimate rotation separately using the velocity-based estimator
            ego_rotation = estimate_ego_rotation(velocities, ["unknown"] * len(velocities))

        ego_frame_velocities.append(ego_velocity)
        ego_frame_rotations.append(ego_rotation)
    
    # smooth the ego velocities using exponential moving average
    ego_frame_velocities_smoothed = smooth_ego_velocity(np.array(ego_frame_velocities))
    # smooth ego rotations using exponential moving average and deadband suppression
    ego_frame_rotations = np.array(ego_frame_rotations)

    # --- EMA smoothing ---
    rot_smoothed = np.zeros_like(ego_frame_rotations)
    rot_smoothed[0] = ego_frame_rotations[0]
    alpha_rot = 0.1   # stronger smoothing

    for i in range(1, len(ego_frame_rotations)):
        rot_smoothed[i] = alpha_rot * ego_frame_rotations[i] + (1 - alpha_rot) * rot_smoothed[i - 1]

    # --- Deadband to suppress small noisy rotations ---
    rotation_deadband = 0.005
    rot_smoothed[np.abs(rot_smoothed) < rotation_deadband] = 0.0

    ego_frame_rotations = rot_smoothed
    # Estimate the ego moving direction based on the smoothed ego velocity
    ego_frame_moving_directions = [estimate_ego_moving_direction(ego_velocity) 
                                   for ego_velocity in ego_frame_velocities_smoothed]
    
    ego_frame_primitives = np.empty(frame_num, dtype=object)
    for frame_index in range(frame_num):
        ego_frame_primitives[frame_index] = {
            "moving_direction": ego_frame_moving_directions[frame_index],
            "velocity": ego_frame_velocities_smoothed[frame_index],
            "rotation": ego_frame_rotations[frame_index],
            "label": "ego_vehicle",
            "frame": all_frame_filenames[frame_index]
        }
    return ego_frame_primitives, object_frame_primitives
    
def estimate_other_primitives(obj_primitives, ego_primitives):
    # estimate the other primitives based on the ego primitives.
    # For example, estimate the absolute moving direction of the other objects based on the ego moving direction
    
    for frame_index in range(len(ego_primitives)):
        ego_velocity = ego_primitives[frame_index]["velocity"]
        ego_moving_direction = ego_primitives[frame_index]["moving_direction"]
        for obj_id in range(obj_primitives.shape[1]):
            obj_prim = obj_primitives[frame_index, obj_id]
            if obj_prim is not None:
                # Estimate the absolute moving direction of the other object based on the ego moving direction
                relative_velocity = obj_prim["velocity"] - ego_velocity
                absolute_moving_direction = estimate_ego_moving_direction(relative_velocity)
    
    
     
    
    
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
        renderer = fig.canvas.get_renderer()
        buf = renderer.buffer_rgba()
        w, h = int(renderer.width), int(renderer.height)
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = img[:, :, :3]  # Convert RGBA to RGB
        plt.close(fig)
        return img
    
    def create_velocity_charts_img(velocities_x, velocities_z, frame_indices):
        """Create velocity line charts for x and z axes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left subplot: Velocity along X axis (Left/Right)
        ax1.plot(frame_indices, velocities_x, color="orange", label="X Velocity", linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Frame Index', fontsize=12)
        ax1.set_ylabel('Velocity X (units/frame)', fontsize=12)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Velocity along X-axis (Left/Right)', fontsize=12)
        ax1.legend(loc='upper right')
        
        # Right subplot: Velocity along Z axis (Forward/Backward)
        ax2.plot(frame_indices, velocities_z, color="blue", label="Z Velocity", linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_ylabel('Velocity Z (units/frame)', fontsize=12)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Velocity along Z-axis (Forward/Backward)', fontsize=12)
        ax2.legend(loc='upper right')
        
        # Add overall title
        fig.suptitle('Ego Velocity Over Time', fontsize=14)
        plt.tight_layout()
        
        # convert to numpy array
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        buf = renderer.buffer_rgba()
        w, h = int(renderer.width), int(renderer.height)
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = img[:, :, :3]  # Convert RGBA to RGB
        plt.close(fig)
        return img

    def create_rotation_chart_img(rotations, frame_indices):
        """Create ego rotation history chart."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        ax.plot(frame_indices, rotations, color="green", linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Yaw Rate (rad/frame)', fontsize=12)
        ax.set_title('Ego Rotation (Yaw Rate History)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        buf = renderer.buffer_rgba()
        w, h = int(renderer.width), int(renderer.height)
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = img[:, :, :3]
        plt.close(fig)
        return img

    def create_rotation_state_bar_img(rotations, frame_indices):
        """
        Create a square visualization showing ego rotation states across time.
        The bar spans from top to bottom and extends horizontally across frames.

        Colors:
        green = straight
        red   = left turn
        blue  = right turn
        """
        import matplotlib.patches as mpatches

        states = []
        threshold =  0.1

        for r in rotations:
            if r > threshold:
                states.append("right")
            elif r < -threshold:
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
    
    def add_primitive_text_to_frame(frame, velocity, moving_direction, rotation):
        """Add primitive text overlay on top of the frame with black background."""
        frame_with_text = frame.copy()
        
        # Determine the detected direction using median of moving_direction array
        # Use middle threshold value (50% of the array)
        mid_idx = len(moving_direction) // 2
        lr_direction = moving_direction[mid_idx, 0]  # left/right
        fb_direction = moving_direction[mid_idx, 1]  # forward/backward
        
        # Convert direction values to text
        if lr_direction > 0:
            lr_text = "Right"
        elif lr_direction < 0:
            lr_text = "Left"
        else:
            lr_text = "Stationary"
        
        if fb_direction > 0:
            fb_text = "Forward"
        elif fb_direction < 0:
            fb_text = "Backward"
        else:
            fb_text = "Stationary"
        
        rotation_value = rotation
        text_lines = [
            f"Forward/Backward: {fb_text}",
            f"Left/Right: {lr_text}",
            f"Velocity: [{velocity[0]:.2f}, {velocity[2]:.2f}]",
            f"Yaw rate: {rotation_value:.3f} rad/frame"
        ]
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        padding = 10
        line_spacing = 35
        
        # Calculate black background height
        bg_height = padding * 2 + len(text_lines) * line_spacing
        
        # Create black background at the top
        frame_with_text[0:bg_height, :] = 0
        
        # Add text lines
        y_offset = padding + 25
        for text in text_lines:
            cv2.putText(frame_with_text, text, (padding, y_offset), 
                       font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            y_offset += line_spacing
        
        return frame_with_text
    
    visual_frames = []
    velocities_x_list = []
    velocities_z_list = []
    rotations_list = []
    frame_indices_list = []
    
    for frame_index, primitive in tqdm(enumerate(ego_primitives), total=len(ego_primitives)):
        velocity = primitive["velocity"]
        moving_direction = primitive["moving_direction"]
        rotation = primitive.get("rotation", 0.0)

        # Collect velocity data
        velocities_x_list.append(velocity[0])
        velocities_z_list.append(velocity[2])
        rotations_list.append(rotation)
        frame_indices_list.append(frame_index)

        # Load the original frame
        frame_img = utils.load_frame(primitive["frame"], None, "ego", "ego_vehicle")

        # Create frame with primitive text overlay
        frame_with_text = add_primitive_text_to_frame(frame_img, velocity, moving_direction, rotation)

        # Create the direction threshold analysis charts
        direction_chart_img = create_direction_threshold_analysis_img(velocity, moving_direction, frame_index)

        # Create the velocity charts (showing all data up to current frame)
        velocity_chart_img = create_velocity_charts_img(
            velocities_x_list, velocities_z_list, frame_indices_list
        )

        # Create the rotation chart (showing all data up to current frame)
        rotation_chart_img = create_rotation_chart_img(
            rotations_list, frame_indices_list
        )
        # Create the rotation state progress bar
        rotation_state_bar_img = create_rotation_state_bar_img(
            rotations_list, frame_indices_list
        )

        # First row: original frame and frame with text side by side
        # Make sure both frames have the same height
        if frame_img.shape[0] != frame_with_text.shape[0]:
            frame_with_text = cv2.resize(frame_with_text, (frame_img.shape[1], frame_img.shape[0]))

        first_row = np.hstack((frame_img, frame_with_text))

        # Second row: combine direction charts, velocity charts, rotation charts, and rotation state bar side by side
        # Make sure all chart images have the same height
        chart_imgs = [
            direction_chart_img,
            velocity_chart_img,
            rotation_chart_img,
            rotation_state_bar_img
        ]
        chart_heights = [img.shape[0] for img in chart_imgs]
        max_chart_height = max(chart_heights)
        chart_imgs_resized = []
        for img in chart_imgs:
            if img.shape[0] != max_chart_height:
                aspect_ratio = img.shape[1] / img.shape[0]
                new_width = int(max_chart_height * aspect_ratio)
                img = cv2.resize(img, (new_width, max_chart_height))
            chart_imgs_resized.append(img)
        second_row = np.hstack(chart_imgs_resized)

        # Resize second row to match the width of the first row
        if second_row.shape[1] != first_row.shape[1]:
            aspect_ratio = second_row.shape[1] / second_row.shape[0]
            new_height = int(first_row.shape[1] / aspect_ratio)
            second_row = cv2.resize(second_row, (first_row.shape[1], new_height))

        # Combine both rows vertically
        combined_img = np.vstack((first_row, second_row))
        visual_frames.append(combined_img)

    # save the visual frames as an mp4 video, with 2 fps
    video_path = output_path / f"ego_primitives_{len(ego_primitives)}frames.mp4"

    writer = imageio.get_writer(video_path, fps=2, codec="libx264")

    for frame in visual_frames:
        writer.append_data(frame)

    writer.close()

    print(f"Saved ego primitive video: {video_path}")
    
    
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
    # convert matrices to object data
    other_obj_data = matrix2obj_data(obj_matrix)
    
    # frame primitives
    ego_primitives, object_frame_primitives = estimate_ego_primitives(other_obj_data)
    
    # other object primitives according to ego primitives
    other_primitives = estimate_other_primitives(object_frame_primitives, ego_primitives)
    
    
    
    
    # visualize other object primitives
    # visualize_others_primitives(other_obj_data, config.get_output_path("driving_primitive_visualization"))
    
    # visual ego primitives
    visualize_ego_primitives(ego_primitives, config.get_output_path("driving_ego_primitive_visualization"))
    
    print("finished processing primitives.")