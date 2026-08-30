
import argparse
import os
import platform
import re
import sys
import time
from pathlib import Path

from src.exp_roadpp import config 
from src.exp_roadpp import utils_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the VWSR pipeline"
    )

    # Add arguments for the pipeline
    parser.add_argument("--data_num", type=str, default="all", help="Number of data to process, default is all")

    parser.add_argument("--dataset", type=str, default="bdd100k", choices=["bdd100k"])

    parser.add_argument("--exp", type=str, default="debug", help="Experiment name, default is debug")

    parser.add_argument("--machine", type=str, default="ml-pulsar", help="Machine name, default is ml-pulsar")

    parser.add_argument("--device", type=str, default="cpu", help="Device to use, default is cpu")
    args = parser.parse_args()

    # load experiment configuration
    exp_config_path = config.exp_config_path / f"{args.exp}.json"
    for key, value in utils_data.load_json(exp_config_path).items():
        setattr(args, key, value)


    return args