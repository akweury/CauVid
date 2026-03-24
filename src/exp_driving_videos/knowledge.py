import numpy as np


primitive_lists = [
    
        # object label
        "label", # object label (e.g., car, pedestrian, cyclist)
        "bbox", # object bounding box (x_min, y_min, x_max, y_max)
        "frame", # frame image file name 
        # ego motion
        "vx_ego", # ego velocity along x axis (left/right)
        "vz_ego", # ego velocity along z axis (forward/backward)
        "w_ego", # ego rotation (yaw rate)
        
        # object velocity
        "vx_obj", # object velocity along x axis (left/right) in the camera frame
        "vz_obj", # object velocity along z axis (forward/backward) in the camera frame
        "vx_obj_w", # object velocity along x axis (left/right) with ego motion removed, in the world frame
        "vz_obj_w", # object velocity along z axis (forward/backward) with ego motion removed, in the world frame
        
        # relative position        
        "dx_obj", # displacement along x axis (left/right) relative to ego motion
        "dz_obj", # displacement along z axis (forward/backward) relative to ego motion
        
        # derived primitives
        "d_obj",  # absolute displacement of the object relative to ego (sqrt(dx_obj^2 + dz_obj^2))
        "closing_speed", # the speed at which the object is approaching or separating from the ego (negative means approaching, positive means separating)
]

def init_obj_primitives(position, label, bbox, frame):
    obj_primitives = {}
    for primitive in primitive_lists:
        obj_primitives[primitive] = None
        
    if position is not None:
        obj_primitives['dx_obj'] = position[0]
        obj_primitives['dz_obj'] = position[2]
        obj_primitives['d_obj'] = np.sqrt(position[0]**2 + position[2]**2)    
    
    if label is not None:
        obj_primitives['label'] = label
    if bbox is not None:
        obj_primitives['bbox'] = bbox
    if frame is not None:
        obj_primitives['frame'] = frame
        
    return obj_primitives

def position2direction(positions):
    # Placeholder for the actual implementation of converting positions to moving directions
    # This function should take the positions and calculate the moving direction based on the change in position over time
    # positions: a list of x,y,z positions over time.
    # x axis: negative on the left, positive on the right
    # z axis: positive in the front, negative in the back
    # left: negative x, right: positive x
    side_direction = np.sign(positions[:, 0])  # -1 for left, +1 for right, 0 for no movement in x direction  
    return side_direction



def position2velocity(positions):
    # Placeholder for the actual implementation of converting positions to velocity
    # This function should take the positions and calculate the velocity based on the change in position over time
    # the output should be the same shape as positions, but with the velocity values instead of position values.
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]  # Calculate velocity
    velocity[0] = velocity[1]  # Set the first velocity to be the same as the second one (or you can set it to zero)
    return velocity

def ego_velocity_median_strategy(vx, vz):
    # median strategy: assume most objects are static
    ego_vx = -np.median(vx)
    ego_vz = -np.median(vz)
    return ego_vx, ego_vz


def ego_yaw_rate_median_strategy(vx, vz):
    # median strategy: assume most objects are static
    w_ego = -np.median(vx / (vz + 1e-5))  # Add a small value to avoid division by zero
    return w_ego

def estimate_obj_world_velocity(vx_obj, vz_obj, vx_ego, vz_ego,w_ego, dx_obj, dz_obj):
    """
    estimate object velocity in the world frame by adding ego velocity and 
    the effect of ego rotation to the object velocity in the camera frame.
    """
    vx_obj_w = vx_obj + vx_ego - w_ego * dz_obj
    vz_obj_w = vz_obj + vz_ego + w_ego * dx_obj
    
    return vx_obj_w, vz_obj_w



def estimate_closing_speed(vx_obj_w, vz_obj_w, vx_ego, vz_ego, dx_obj, dz_obj):    
    relative_velocity = np.array([vx_obj_w - vx_ego, vz_obj_w - vz_ego])
    relative_position = np.array([dx_obj, dz_obj])
    distance = np.linalg.norm(relative_position) + 1e-5  # Add a small value to avoid division by zero
    unit_vector = relative_position / distance
    closing_speed = -np.dot(relative_velocity, unit_vector)
    return closing_speed
