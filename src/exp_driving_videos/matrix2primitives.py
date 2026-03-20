
import numpy as np 
import matplotlib.pyplot as plt
import os 
import imageio
import cv2
import torch
from tqdm import tqdm

import config 
from src.exp_driving_videos import exp_driving_utils as utils
from src.exp_driving_videos import knowledge 

def obj_centric2frame_centric(matrix):
    """
    return a 2d matrix with frame index as the rows and object id as the columns,
    """
    total_obj_num = len(matrix)
    frame_num = max([max(primitives["frame_indices"]) for primitives in matrix.values()]) + 1  # Get the total number of frames based on the maximum frame index    
    obj_primitives = np.empty((frame_num, total_obj_num), dtype=object)  # Create an empty array to hold frame primitives
    obj_new_id_mapping = {obj_id: idx for idx, obj_id in enumerate(matrix.keys())}
    
    # Iterate through each frame and populate the obj_primitives matrix
    for frame_index in range(frame_num):
        for obj_id, obj_data in matrix.items():
            if frame_index in obj_data.get('frame_indices', []):
                idx = obj_data['frame_indices'].index(frame_index)
                position = obj_data.get('position', [None] * frame_num)[idx]  # Get the position for this frame
                label = obj_data.get('label', 'unknown')  # Get the label, default to 'unknown' if not present
                bbox = obj_data.get('bboxes', [None] * frame_num)[idx]  # Get the bounding box for this frame
                frame = obj_data.get('frames', [None] * frame_num)[idx]  # Get the frame for this frame index
                
                # all the keys in the primitive_lists should be included in the output, even if some of them are None or not available, to ensure the consistency of the output format.
                obj_primitives[frame_index, obj_new_id_mapping[obj_id]] = knowledge.init_obj_primitives(
                    position=position,
                    label=label,
                    bbox=bbox,
                    frame=frame
                )
    return obj_primitives

def estimate_vx_vz_obj(obj_primitives):
    """ 
    Estimate each object velocity based on the change in position over time.
    
    """
    obj_primitives_estimated = np.empty_like(obj_primitives)
    for obj_id in range(obj_primitives.shape[1]):
        obj_in_frame_mask = [obj_primitives[frame_index, obj_id] is not None for frame_index in range(obj_primitives.shape[0])]
        obj_positions = torch.tensor([[obj_primitive['dx_obj'],obj_primitive['dz_obj']] for obj_primitive in obj_primitives[:, obj_id] if obj_primitive is not None ])
        obj_velocities = knowledge.position2velocity(obj_positions.numpy())
        # smooth the velocities using a simple moving average filter
        window_size = 5
        if len(obj_velocities) >= window_size:
            obj_velocities_smoothed = np.convolve(obj_velocities[:, 0], np.ones(window_size)/window_size, mode='same')
            obj_velocities_smoothed_z = np.convolve(obj_velocities[:, 1], np.ones(window_size)/window_size, mode='same')
            obj_velocities_smoothed = np.stack([obj_velocities_smoothed, obj_velocities_smoothed_z], axis=1)
        for index in range(len(obj_velocities_smoothed)):
            frame_index = [i for i, x in enumerate(obj_in_frame_mask) if x][index]
            if obj_primitives[frame_index, obj_id] is not None:
                obj_primitives_estimated[frame_index, obj_id] = obj_primitives[frame_index, obj_id].copy()                
                obj_primitives_estimated[frame_index, obj_id]['vx_obj'] = obj_velocities_smoothed[index, 0]
                obj_primitives_estimated[frame_index, obj_id]['vz_obj'] = obj_velocities_smoothed[index, 1]
            else:
                obj_primitives_estimated[frame_index, obj_id] = None
    
    return obj_primitives_estimated

def estimate_ego_motion(obj_primitives, window_size=5):
    """ 
    Estimate the ego velocity and rotation based on the relative velocity of the other objects in the frame.
    The estimation is done separately for each frame, and then smoothed across frames to reduce noise.
    """
    obj_primitives_new = np.empty_like(obj_primitives)
    vx_ego_list = []
    vz_ego_list = []
    w_ego_list = []
    
    for frame_index in range(obj_primitives.shape[0]):
        frame_obj_velocities = []
        for obj_prim in obj_primitives[frame_index]:
            if obj_prim is None:
                continue
            vx_obj = obj_prim.get("vx_obj", None)
            vz_obj = obj_prim.get("vz_obj", None)
            if vx_obj is None or vz_obj is None:
                continue
            vel = np.array([vx_obj, 0, vz_obj])
            frame_obj_velocities.append(vel)
        frame_obj_velocities = np.array(frame_obj_velocities)
        
        if frame_obj_velocities.size == 0:
            vx_ego = 0.0
            vz_ego = 0.0
            w_ego = 0.0
        else:
            vx = frame_obj_velocities[:, 0]
            vz = frame_obj_velocities[:, 2]
            vx_ego, vz_ego = knowledge.ego_velocity_median_strategy(vx, vz)
            # estimate rotation separately using the velocity-based estimator
            w_ego = knowledge.ego_yaw_rate_median_strategy(vx, vz)
        
        vx_ego_list.append(vx_ego)
        vz_ego_list.append(vz_ego)
        w_ego_list.append(w_ego)
        
        # update vx_ego, vx_ego, w_ego to all the objects in the frame, since they share the same ego motion
        for i in range(obj_primitives.shape[1]):
            if obj_primitives[frame_index, i] is not None:
                obj_primitives_new[frame_index, i] = obj_primitives[frame_index, i].copy()
                obj_primitives_new[frame_index, i]['vx_ego'] = vx_ego
                obj_primitives_new[frame_index, i]['vz_ego'] = vz_ego
                obj_primitives_new[frame_index, i]['w_ego'] = w_ego

    # smooth the ego velocities and rotations across frames using a simple moving average filter
    ego_velocities = np.array(list(zip(vx_ego_list, vz_ego_list)))  
    ego_w = np.array(w_ego_list)
    ego_velo_smoothed_x = np.convolve(ego_velocities[:, 0], np.ones(window_size)/window_size, mode='same')
    ego_velo_smoothed_z = np.convolve(ego_velocities[:, 1], np.ones(window_size)/window_size, mode='same')
    ego_w_smoothed = np.convolve(ego_w, np.ones(window_size)/window_size, mode='same')
    
    ego_motion = np.stack([ego_velo_smoothed_x, ego_velo_smoothed_z, ego_w_smoothed], axis=1)
    # update the smoothed ego velocities and rotations back to the obj_primitives_new
    for frame_index in range(obj_primitives_new.shape[0]):
        for i in range(obj_primitives_new.shape[1]):
            if obj_primitives_new[frame_index, i] is not None:
                obj_primitives_new[frame_index, i]['vx_ego'] = ego_velo_smoothed_x[frame_index]
                obj_primitives_new[frame_index, i]['vz_ego'] = ego_velo_smoothed_z[frame_index]
                obj_primitives_new[frame_index, i]['w_ego'] = ego_w_smoothed[frame_index]
    
    return obj_primitives_new, ego_motion 

def estimate_obj_world_primitives(obj_primitives, window_size=3):
    """ 
    Estimate the object primitives in the world frame vx_obj_w and vz_obj_w.
    The ego motion includes both the ego velocity and ego rotation. 
    For velocity, we can simply add the ego velocity to the object's velocity to get the velocity in the world frame.    
    """
    obj_primitives_new = np.empty_like(obj_primitives)
    for frame_index in range(obj_primitives.shape[0]):
        for obj_id in range(obj_primitives.shape[1]):
            if obj_primitives[frame_index, obj_id] is not None:
                obj_prim = obj_primitives[frame_index, obj_id]
                vx_ego = obj_prim.get("vx_ego", 0.0)
                vz_ego = obj_prim.get("vz_ego", 0.0)
                w_ego = obj_prim.get("w_ego", 0.0)
                vx_obj = obj_prim.get("vx_obj", 0.0)
                vz_obj = obj_prim.get("vz_obj", 0.0)
                dz_obj = obj_prim.get("dz_obj", 0.0)
                dx_obj = obj_prim.get("dx_obj", 0.0)
                
                if vx_ego is not None and vz_ego is not None and w_ego is not None:
                    vx_obj_w, vz_obj_w = knowledge.estimate_obj_world_velocity(
                        vx_obj, vz_obj, vx_ego, vz_ego, w_ego, dx_obj, dz_obj)
                    obj_primitives_new[frame_index, obj_id] = {
                        **obj_prim,
                        "vx_obj_w": vx_obj_w,
                        "vz_obj_w": vz_obj_w
                    }
    # smooth the world velocities across frames using a simple moving average filter
    for obj_id in range(obj_primitives_new.shape[1]):
        obj_in_frame_mask = [obj_primitives_new[frame_index, obj_id] is not None for frame_index in range(obj_primitives_new.shape[0])]
        obj_velocities_w = torch.tensor([[obj_primitive['vx_obj_w'],obj_primitive['vz_obj_w']] for obj_primitive in obj_primitives_new[:, obj_id] if obj_primitive is not None ])
        if len(obj_velocities_w) >= window_size:
            obj_velocities_w_smoothed_x = np.convolve(obj_velocities_w[:, 0], np.ones(window_size)/window_size, mode='same')
            obj_velocities_w_smoothed_z = np.convolve(obj_velocities_w[:, 1], np.ones(window_size)/window_size, mode='same')
            obj_velocities_w_smoothed = np.stack([obj_velocities_w_smoothed_x, obj_velocities_w_smoothed_z], axis=1)
            for index in range(len(obj_velocities_w_smoothed)):
                frame_index = [i for i, x in enumerate(obj_in_frame_mask) if x][index]
                if obj_primitives_new[frame_index, obj_id] is not None:
                    obj_primitives_new[frame_index, obj_id]['vx_obj_w'] = obj_velocities_w_smoothed[index, 0]
                    obj_primitives_new[frame_index, obj_id]['vz_obj_w'] = obj_velocities_w_smoothed[index, 1]
 
    return obj_primitives_new 
    
    
    
    
def estimate_closing_speed(obj_primitives):
    """ 
    Estimate the closing speed between the ego and the object, which is the speed at which the object is approaching or separating from the ego.
    The closing speed can be calculated as the negative of the dot product of the relative velocity vector 
    and the unit vector pointing from the ego to the object.
    """
    obj_primitives_new = np.empty_like(obj_primitives)
    for frame_index in range(obj_primitives.shape[0]):
        for obj_id in range(obj_primitives.shape[1]):
            if obj_primitives[frame_index, obj_id] is not None:
                obj_prim = obj_primitives[frame_index, obj_id]
                vx_ego = obj_prim.get("vx_ego", 0.0)
                vz_ego = obj_prim.get("vz_ego", 0.0)
                vx_obj_w = obj_prim.get("vx_obj_w", 0.0)
                vz_obj_w = obj_prim.get("vz_obj_w", 0.0)
                dx_obj = obj_prim.get("dx_obj", 0.0)
                dz_obj = obj_prim.get("dz_obj", 0.0)
                
                if vx_ego is not None and vz_ego is not None and vx_obj_w is not None and vz_obj_w is not None:
                    closing_speed = knowledge.estimate_closing_speed(vx_obj_w, vz_obj_w, vx_ego, vz_ego, dx_obj, dz_obj)                   
                    obj_primitives_new[frame_index, obj_id] = {
                        **obj_prim,
                        "closing_speed": closing_speed
                    }
    return obj_primitives_new



def visualize_ego_primitives(obj_primitives, output_path, video_id=None):
    os.makedirs(output_path, exist_ok=True)
    if video_id is None:
        video_path = output_path / f"ego_primitives_{config.driving_demo_video_id}.mp4"
    else:
        video_path = output_path / f"ego_primitives_{video_id}.mp4"

    # if video already exists, skip the visualization process
    if video_path.exists():
        print(f"Video {video_path} already exists, skipping visualization.")
        return

    
    visual_frames = []
    velocities_x_list = []
    velocities_z_list = []
    rotations_list = []
    frame_indices_list = []
    
    for frame_index, frame_objs_primitive in tqdm(enumerate(obj_primitives), total=len(obj_primitives)):
        frame_indices_list.append(frame_index)    
        
        # first non None object idx
        non_obj_index = next((i for i, obj in enumerate(frame_objs_primitive) if obj is not None), None)
        # frame image
        
        frame_img = utils.load_frame(frame_objs_primitive[non_obj_index]["frame"])
        
        # create ego x velocity chart
        velocities_x_list.append(frame_objs_primitive[non_obj_index].get("vx_ego", 0.0))
        vx_ego_img = utils.create_line_chart_img(frame_indices_list, velocities_x_list, 
                                                title="Ego Velocity X History", xlabel="Frame Index", ylabel="Velocity X (units/frame)")
        # create ego z velocity chart
        velocities_z_list.append(frame_objs_primitive[non_obj_index].get("vz_ego", 0.0))
        vz_ego_img = utils.create_line_chart_img(frame_indices_list, velocities_z_list,
                                                title="Ego Velocity Z History", xlabel="Frame Index", ylabel="Velocity Z (units/frame)")
        
        # create ego rotation chart
        rotations_list.append(frame_objs_primitive[non_obj_index].get("w_ego", 0.0))
        ego_w_img = utils.create_line_chart_img(frame_indices_list, rotations_list,
                                                title="Ego Yaw Rate History",xlabel="Frame Index", ylabel="Yaw Rate (rad/frame)")
        # create rotation state progress bar chart
        ego_w_state_bar_img = utils.create_state_bar_chart_img(rotations_list,state_threshold=0.1)

        first_row = frame_img

        # Second row: combine direction charts, velocity charts, rotation charts, rotation state bar, and world speed chart side by side
        # Make sure all chart images have the same height
        chart_imgs = [vx_ego_img, vz_ego_img, ego_w_img, ego_w_state_bar_img]
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
    
    



def matrix2primitives(matrix, video_id=None, visualize_ego=False, visualize_others=False, save_primitives=False):
    # check if primitive data is existed or not, if existed, load the primitive data and return directly to save time, otherwise, process the matrix to get the primitives.
    
    def save_primitives_to_file(obj_primitives, ego_motion, output_path, video_id=None):
        os.makedirs(output_path, exist_ok=True)
        if video_id is None:
            file_path = output_path / f"obj_primitives_{config.driving_demo_video_id}.pkl"
        else:
            file_path = output_path / f"obj_primitives_{video_id}.pkl"
        utils.save_pkl_file(file_path, {"obj_primitives": obj_primitives, "ego_motion": ego_motion})
        print(f"Saved primitives to file: {file_path}")
    
    def load_primitive_files(file_path):
        data = utils.load_matrix(file_path)
        obj_primitives = data.get("obj_primitives", None)
        ego_motion = data.get("ego_motion", None)
        print(f"Loaded primitives from file: {file_path}")
        return obj_primitives, ego_motion

    primitive_file_path = config.get_output_path("pipeline_output") / f"obj_primitives_{video_id}.pkl"
    if primitive_file_path.exists():
        obj_primitives, ego_motion = load_primitive_files(primitive_file_path)  # This will print the loaded primitive data for debugging
        print(f"Primitive file already exists for video {video_id}, loading from file: {primitive_file_path}")
        return obj_primitives, ego_motion
    
    # convert matrices to frame-centric object primitives    
    obj_primitives = obj_centric2frame_centric(matrix)
    
    # estimate velocity of other objects in the camera frame based on the change in position over time, 
    # and smooth the velocity using a simple moving average filter
    obj_primitives = estimate_vx_vz_obj(obj_primitives)
    
    # estimate ego motion, including ego velocity and ego rotation, 
    # based on the relative velocity of the other objects in the frame,
    obj_primitives, ego_motion = estimate_ego_motion(obj_primitives)
    
    # estimate other world primitives, including object velocity in the world frame by adding ego velocity 
    # and the effect of ego rotation to the object velocity in the camera frame, 
    # and then smooth the world velocity across frames using a simple moving average filter.
    obj_primitives = estimate_obj_world_primitives(obj_primitives)
    
    # estimate closing speed
    obj_primitives = estimate_closing_speed(obj_primitives)
    
    if visualize_ego:
        # visual ego primitives
        visualize_ego_primitives(obj_primitives, config.get_output_path("driving_ego_primitive_visualization"), video_id)
    
    if save_primitives:
        save_primitives_to_file(obj_primitives,ego_motion, config.get_output_path("pipeline_output"), video_id)
        
    return obj_primitives,ego_motion
    
    
if __name__ == "__main__":    
    # load processed matrices
    obj_matrix = utils.load_matrix(config.get_output_path("pipeline_output") / f"smoothed_object_matrices_{config.driving_demo_video_id}.pkl")    
    # convert matrices to frame-centric object primitives    
    obj_primitives = obj_centric2frame_centric(obj_matrix)
    
    # estimate primitives
    obj_primitives = estimate_vx_vz_obj(obj_primitives)
    
    # estimate ego motion
    obj_primitives = estimate_ego_motion(obj_primitives)
    
    # estimate other world primitives
    obj_primitives = estimate_obj_world_primitives(obj_primitives)
    # estimate closing speed
    obj_primitives = estimate_closing_speed(obj_primitives)
    
    # visual ego primitives
    visualize_ego_primitives(obj_primitives,config.get_output_path("driving_ego_primitive_visualization"))
    
    print("finished processing primitives.")