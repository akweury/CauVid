import numpy as np


from exp_roadpp import utils_pipe
from src.exp_roadpp import step01, step02
from src.exp_roadpp import config



def main():
    args = utils_pipe.parse_args()
    # step 0: system check, and set up the environment
    args = config.step_0_setup(args)
    # step 1: detect objects masks, labels, depths in
    # each frame of the video, cache the results as intermediate files
    step01_input = config.get_step_01_input(args)
    step01.main(step01_input)

    # step 2: estimate the 3D pose of each object in each frame.
    step02_input = config.get_step_02_input(args)
    step02.main(step02_input)
    return   


if __name__ == "__main__":
    print("\n------- Start the pipeline -------\n")
    res = main()
    print("\n ------- Program Finished! ------ \n")