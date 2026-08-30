
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import json

from src.exp_roadpp import utils_data
from src.exp_roadpp.external_roadpp_repo import convert_waymo_to_coco 


def main(input_data):
    print("\n------- Step 01 -------\n")
    video_dir = input_data["video_dir"]
    convert_waymo_to_coco.main(input_data["video_dir"])
    print("\n--------- Step 01 Done ---------------\n")

    
