import numpy as np


from src.exp_roadpp import utils_pipe
from src.exp_roadpp import step01, step02, step03, step04, step05, step05_causal, step05_rules, step05_motif, step05_layer, step06, step06_layer, step_dataset, step_baselines
from src.exp_roadpp import step05_state, step06_state
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

    step_dataset_input = config.get_step_dataset_input(args)
    step_dataset.main(step_dataset_input)
    
    # step 3: use a language model to generate anchor clauses
    step03_input = config.get_step_03_input(args)
    step03.main(step03_input)
    
    # anchor clauses pruning (step 4)
    step04_input = config.get_step_04_input(args)
    step04.main_v2(step04_input)

    # clause extension (step 5)
    step05_input = config.get_step_05_input(args)
    step05.main(step05_input)
    # causal pruning of the learned rule net (conditional independence tests per clause atom)
    step05_causal.main(step05_input)
    # rule set selection: every rule precise, the set covering the events (greedy set cover)
    step05_rules.main(step05_input)
    # two-level rule net: 2-frame motifs, 10-frame rules, every clause pushed to a minimum precision
    step05_motif.main(step05_input)
    # layered rule net: predicts the full atom state of each layer of frames from the K before it
    if args.layered_rules == "True":
        step05_layer.main(step05_input)
    # state forecast: the GT label state of every agent 2 s ahead, every frame (StateMLP)
    if args.state_forecast == "True":
        step05_state.main(step05_input)

    # estimation and evaluation of the extended clauses over the val videos
    step06_input = config.get_step_06_input(args)
    step06.main(step06_input)
    if args.layered_rules == "True":
        step06_layer.main(step06_input)
    if args.state_forecast == "True":
        step06_state.main(step06_input)

    raise NotImplementedError()
    # run baseline methods
    step_baselines_input = config.get_step_baselines_input(args)
    step_baselines.main(step_baselines_input)


if __name__ == "__main__":
    print("\n------- Start the pipeline -------\n")
    res = main()
    print("\n ------- Program Finished! ------ \n")