import numpy as np


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