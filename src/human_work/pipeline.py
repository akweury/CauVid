# this file is coded without using AI.

"""
# the pipeline has the following steps:
# step 1: detect objects masks, labels, depths in 
# each frame of the video, cache the results as intermediate files
# step 2: estimate the 3D pose of each object in each frame.

# step 3: rebuild the world state of the camera and the objects

# step 4: reasoning the conflict of the world states based 
# on the different fact sources

# step 5: reestimate the world state with less conflicts

# step 6: comparing with the ground truth,
# check the accuracy of the world state estimation.
"""

import numpy as np 
