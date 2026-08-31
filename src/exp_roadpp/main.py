import numpy as np


import utils_pipe
import step01, step02, step03, step04
import config



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

    # step 3: take trajectories as input, output symbolic facts
    step03_input = config.get_step_03_input(args)
    step03.main(step03_input)

    # step 4: causal reasoning
    step_04_input = config.get_step_04_input(args)
    step04.main(step_04_input)

    return   


if __name__ == "__main__":
    print("\n------- Start the pipeline -------\n")
    res = main()
    print("\n ------- Program Finished! ------ \n")