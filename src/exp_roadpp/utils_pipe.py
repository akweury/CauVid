
import argparse
import os
import platform
import re
import sys
import time
from pathlib import Path

from src.exp_roadpp import config 
from src.exp_roadpp import utils_data

def parse_args(parser=None):
    if parser is None:
        parser = get_parser()

    args = parser.parse_args()

    # load experiment configuration: its values replace the parser defaults, but an option
    # given explicitly on the command line (e.g. --num_epochs 50) keeps its value.
    given = {action.dest for action in parser._actions for opt in action.option_strings
             if any(arg == opt or arg.startswith(opt + "=") for arg in sys.argv[1:])}
    exp_config_path = config.exp_config_path / f"{args.exp}.json"
    for key, value in utils_data.load_json(exp_config_path).items():
        if key not in given:
            setattr(args, key, value)


    return args

def get_parser():
    parser = argparse.ArgumentParser(
        description="Run the VWSR pipeline"
    )

    # Add arguments for the pipeline
    parser.add_argument("--data_num", type=str, default="full", help="Number of data to process, default is full")

    parser.add_argument("--dataset", type=str, default="roadpp", choices=["bdd100k", "roadpp"])

    parser.add_argument("--exp", type=str, default="debug", help="Experiment name, default is debug")

    parser.add_argument("--machine", type=str, default="ml-pulsar", help="Machine name, default is ml-pulsar")
    parser.add_argument("--grid", type=str, default="True",
                        help="Step 05: grid atoms of the 2D boxes (grid.py), default True; False for the no-grid baseline")
    parser.add_argument("--motion", type=str, default="True",
                        help="Step 05: motion atoms of the own 2D box (motion.py), default True; False for the ablation")
    parser.add_argument("--ego_motion", type=str, default="False",
                        help="Step 05: ego motion atoms read from the scene boxes (ego_motion.py), default False; True for the ablation")
    parser.add_argument("--require_self", type=str, default="True",
                        help="Step 05: every rule must hold an own-agent atom (TemporalRuleNet require_self), default True")
    # Ablation settings (`ablation.py` runs the whole suite); the defaults are the main model.
    parser.add_argument("--anchor_source", type=str, default="causal", choices=["causal", "all", "random", "none"],
                        help="Step 05: anchor clauses seeding the rules: causal (step 04 filter, default), all (every "
                             "LLM anchor, no causal filter), random (random LLM anchors, as many as causal), none")
    parser.add_argument("--anchor_subset_seed", type=int, default=0, help="Seed of --anchor_source random")
    parser.add_argument("--num_free_rules", type=int, default=None,
                        help="Step 05: unseeded rules (default 128; with --anchor_source none 128 + the causal seeds)")
    parser.add_argument("--truncate_prob", type=float, default=0.5,
                        help="Step 05: share of training frames with a truncated history (partial facts), default 0.5")
    parser.add_argument("--num_slots", type=int, default=3, help="Step 05: memory slots of 10 frames, default 3")
    parser.add_argument("--roles", type=str, default="", help="Step 05: agent roles, e.g. ahead,near_ped (default none)")
    parser.add_argument("--geometry", type=str, default="False", help="Step 05: geometry atoms, default False")
    parser.add_argument("--type_balance", type=str, default="False", help="Step 05: event type balance, default False")
    parser.add_argument("--eval_min_context", type=int, default=0,
                        help="Step 06/07: score from this frame on at least (0: the model's memory)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Resume this run (folder output/<exp>/runs/<run_id>) instead of starting a new one, "
                             "e.g. to redo step 06 of a trained model; default: a new run folder")

    parser.add_argument("--device", type=str, default="cpu", help="Device to use, default is cpu")
    
    parser.add_argument("--learning", type=str, default="False", help="Whether to perform rule learning, default is False")
    parser.add_argument("--baselines", type=str, default="False", help="Whether to run baselines, default is False")
    parser.add_argument("--num_epochs", type=int, default=200, help="Training epochs of the step 05 TemporalRuleNet, default is 1000")
    parser.add_argument("--layered_rules", type=str, default="True", help="Whether to also train the layered rule net (step05_layer) after step 05, default is True")
    parser.add_argument("--state_forecast", type=str, default="True",
                        help="Train the StateMLP (step05_state: the GT label state of every agent 2 s ahead, every frame) "
                             "and evaluate it with review videos (step06_state), default True")

    parser.add_argument("--paper", type=str, default="True",
                        help="Step 07 (step07_paper): every onset model at fixed false-alarm budgets, thresholds on val, "
                             "and the test events each detects (paper_metrics.json, paper_events.csv), default True")

    parser.add_argument("--skip_baselines_03", type=str, default="True", help="Whether to skip baselines, default is True")

    # Causal-discovery substep (formerly step 04), now run inside step 03 to filter
    # coarse-pruned clauses with a causal graph before the extension loop.
    parser.add_argument("--use_causal_filter", type=str, default="True",
                         help="Whether to filter coarse-pruned clauses with a causal graph (former step 04), default is True")
    parser.add_argument("--min_support_features", type=int, default=1,
                         help="Minimum number of occurrences for a feature to be kept in the causal-discovery dataset")
    parser.add_argument("--cd_method", type=str, default="boss", choices=["pc", "fci", "ges", "boss", "grasp"],
                         help="Causal discovery method")
    parser.add_argument("--cd_stat", type=str, default="bic", choices=["fisherz", "chisq", "gsq", "bic", "bdeu"],
                         help="CI test for pc / fci, score for ges / boss / grasp")
    parser.add_argument("--cd_alpha", type=float, default=0.01, help="Significance level of the CI tests (pc / fci only)")
    parser.add_argument("--cd_depth", type=int, default=None, help="Maximum conditioning-set size (pc / fci only), default unlimited")
    parser.add_argument("--graph_strategy", type=str, default="adjacent", choices=["adjacent"],
                         help="How the causal graph filters the clauses")

    return parser