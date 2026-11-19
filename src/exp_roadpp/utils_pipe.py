
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
    parser.add_argument("--motion", type=str, default="False",
                        help="Step 05: motion atoms of the own 2D box (motion.py), default False (off since 2026-11-15: "
                             "no effect on medium or full); True for the ablation or to resume a run trained with them")
    parser.add_argument("--ego_motion", type=str, default="False",
                        help="Step 05: ego motion atoms read from the scene boxes (ego_motion.py), default False; True for the ablation")
    parser.add_argument("--ego_flow", type=str, default="False",
                        help="Step 05: ego yaw / speed atoms read from the camera image (ego_flow.py), default False")
    parser.add_argument("--slot_ranges", type=str, default="False",
                        help="Step 05: rule bodies over a span of slots a..b (variable step durations), default False")
    parser.add_argument("--predict", type=str, default="False",
                        help="Step 05: prediction atoms (predict.py): when the ego reaches a pedestrian / cyclist "
                             "and whether their paths cross, from the boxes; default False")
    parser.add_argument("--durations", type=str, default="False",
                        help="Step 05: duration atoms, how long a label has held (duration.py), default False")
    parser.add_argument("--duration_labels", type=str, default="",
                        help="Step 05: comma-separated labels with duration atoms (default duration.LABELS)")
    parser.add_argument("--ego_flow_workers", type=int, default=16,
                        help="Processes decoding the videos for the ego flow cache (CPU)")
    parser.add_argument("--require_self", type=str, default="True",
                        help="Step 05: every rule must hold an own-agent atom (TemporalRuleNet require_self), default True")
    # Ablation settings (`ablation.py` runs the whole suite); the defaults are the main model.
    parser.add_argument("--anchor_source", type=str, default="causal", choices=["causal", "all", "random", "none"],
                        help="Step 05: anchor clauses seeding the rules: causal (step 04 filter, default), all (every "
                             "LLM anchor, no causal filter), random (random LLM anchors, as many as causal), none")
    parser.add_argument("--anchor_subset_seed", type=int, default=0, help="Seed of --anchor_source random")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Step 05: learning curve, train on this share of the train videos (nested subsets; "
                             "val, test and the head types unchanged; each epoch visits the share 1 / share times, the "
                             "same steps per epoch), default 1")
    parser.add_argument("--train_subset_seed", type=int, default=0, help="Seed of the --train_fraction subset")
    parser.add_argument("--num_free_rules", type=int, default=None,
                        help="Step 05: unseeded rules (default 128; with --anchor_source none 128 + the causal seeds)")
    parser.add_argument("--per_head_rules", type=str, default="False",
                        help="Step 05: every rule serves one head (rule_copies per anchor body, head and slot, "
                             "free_rules_per_head free rules per learned head), default False")
    parser.add_argument("--rule_copies", type=int, default=2,
                        help="Step 05 (--per_head_rules): rules per (anchor body, head, slot), default 2")
    parser.add_argument("--free_rules_per_head", type=int, default=4,
                        help="Step 05 (--per_head_rules): free rules per learned head, default 4")
    parser.add_argument("--copy_input_share", type=float, default=0.5,
                        help="Step 05 (--per_head_rules): share of the inputs the extra copies may learn, default 0.5")
    parser.add_argument("--rule_lr", type=float, default=None,
                        help="Step 05: lr of the rule net's head weights / biases (default 0.01; the gates keep gate_lr)")
    parser.add_argument("--l0_lambda", type=float, default=None,
                        help="Step 05: weight of the rule net's sparsity loss (mask L1 + head weight L1), default 1e-3")
    parser.add_argument("--init_log_alpha", type=float, default=None,
                        help="Step 05: initial gate logit of the rule net's masks, default -11")
    parser.add_argument("--train_seed", type=int, default=0,
                        help="Step 05: seed of the init and the frame sampling of the rule net and the MLP, default 0")
    parser.add_argument("--truncate_prob", type=float, default=0.5,
                        help="Step 05: share of training frames with a truncated history (partial facts), default 0.5")
    parser.add_argument("--num_slots", type=int, default=3, help="Step 05: memory slots of 10 frames, default 3")
    parser.add_argument("--roles", type=str, default="", help="Step 05: agent roles, e.g. ahead,near_ped,xing_ped (default none)")
    parser.add_argument("--geometry", type=str, default="False", help="Step 05: geometry atoms, default False")
    parser.add_argument("--type_balance", type=str, default="False", help="Step 05: event type balance, default False")
    parser.add_argument("--review_videos", type=str, default="False",
                        help="Step 06: render the review videos (review/, state_review/; ~1.4 GB each on full, slow), "
                             "default False; True to render them (e.g. later for one run: --run_id <run> --review_videos True)")
    parser.add_argument("--event_clips", type=str, default="False",
                        help="Step 08: short clips around single test events and false alarms for a human review of "
                             "predictability (step06_output/clips, packs/clips_<run>.tar.gz), default False; "
                             "e.g. for a finished run: --run_id <run> --event_clips True")
    parser.add_argument("--eval_min_context", type=int, default=0,
                        help="Step 06/07: score from this frame on at least (0: the model's memory)")
    parser.add_argument("--split", type=str, default="random", choices=["random", "stratified"],
                        help="Video split of step_dataset: random (the old one, exps medium / full) or stratified by "
                             "the onsets of every head (exps medium_s / full_s); set in exp_config/<exp>.json")
    parser.add_argument("--min_vehicle_height", type=float, default=0.0,
                        help="step_dataset: drop the frames where a vehicle's (Car, MedVeh, LarVeh, Bus, EmVeh) GT box "
                             "is below this share of the frame height (far vehicles); 0 keeps all; set in exp_config (*_sn)")
    parser.add_argument("--split_from", type=str, default=None,
                        help="step_dataset: take the video split of this exp (e.g. full_s) instead of a new one")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Resume this run (folder output/<exp>/runs/<run_id>) instead of starting a new one, "
                             "e.g. to redo step 06 of a trained model; default: a new run folder")

    parser.add_argument("--device", type=str, default="cpu", help="Device to use, default is cpu")
    
    parser.add_argument("--learning", type=str, default="False", help="Whether to perform rule learning, default is False")
    parser.add_argument("--baselines", type=str, default="False", help="Whether to run baselines, default is False")
    parser.add_argument("--num_epochs", type=int, default=200, help="Training epochs of the step 05 TemporalRuleNet, default is 1000")
    parser.add_argument("--layered_rules", type=str, default="True", help="Whether to also train the layered rule net (step05_layer) after step 05, default is True")
    parser.add_argument("--state_forecast", type=str, default="False",
                        help="Train the StateMLP (step05_state: the GT label state of every agent 2 s ahead, every frame) "
                             "and evaluate it (step06_state), default False (off since 2026-11-15: no clear gain over "
                             "the own-label lookup on medium)")

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