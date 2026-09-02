
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import json

from src.exp_roadpp import utils_data


def split_gt_json_files(gt_json_file, out_dir, data_num):
    # if out_dir has not _gt.json files, create them
    if not any(fname.endswith("_gt.json") for fname in os.listdir(out_dir)):
        # Implement the logic to split the GT JSON file if needed
        print(f"Splitting GT JSON file: {gt_json_file}")
        json_data = utils_data.load_json(gt_json_file)
        counter = 0
        for vid, data in tqdm(json_data['db'].items(), desc="Splitting GT for videos"):
            if "agent_tubes" not in data:
                continue
            json_file_name = os.path.join(out_dir, f"{vid}_gt.json")
            utils_data.save_json({
                "vid": vid,
                "data": data
                }, json_file_name)
            counter += 1

def load_gt_json_file(gt_dir):
    gt_json_dict = {}
    # get all the _gt.json file paths
    for fname in os.listdir(gt_dir):
        if not fname.endswith("_gt.json"):
            continue
        vid = fname.split('_gt.json')[0]
        gt_json_dict[vid] = os.path.join(gt_dir, fname)


    return gt_json_dict

    
def load_od_model(input_data):
    pass

def load_mask_model(input_data):
    pass

def load_depth_model(input_data):
    pass

def load_flow_model(input_data):
    pass

def load_packing_model(input_data):
    pass

def _video_to_frames(video_path, output_dir):
    """
    Convert a video into frames and save them as images in the output directory.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
    Returns:
        All frames path in the output directory.
    """
    if os.path.exists(output_dir):
        output_dir = Path(output_dir)
        frame_paths = sorted(
            path for path in output_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        return frame_paths

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break        
        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()
    return sorted(
        str(path) for path in Path(output_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

def _video_to_low_fps_frames(frame_rate, frame_paths):
    if frame_rate <= 0:
        raise ValueError("Frame rate must be a positive integer.")
    
    low_frame_paths = [
        Path(frame_path) for index, frame_path in enumerate(frame_paths) if index % frame_rate == 0
    ]
    return low_frame_paths

def _frames_to_objects(video_id, frame_rate, frame_paths, obj_dir, od_model):
    pass

def _frames_to_masks(video_id, frame_rate, frame_paths, mask_dir, mask_model, mask_label_top_k):
    pass

def _frames_to_depths(frame_rate, frame_paths, depth_path, depth_model):
    pass

def _frames_to_flows(frame_rate, frame_paths, flow_path, flow_model):
    pass

def _frames_to_records(video_id, frame_rate, frame_paths, depth_path, flow_path, obj_dir, mask_dir, record_dir, packing_model):
    pass





def main(input_data):
    print("\n------- Step 01 -------\n")
    od_model = load_od_model(input_data)
    mask_model = load_mask_model(input_data)
    depth_model = load_depth_model(input_data)
    flow_model = load_flow_model(input_data)
    packing_model = load_packing_model(input_data)
    mask_label_top_k = int(input_data.get("mask_label_top_k", 3))
    all_video_ids = input_data["video_ids"]
    all_video_paths = input_data["video_path"]
    all_frame_paths = input_data["frame_path"]
    all_depth_paths = input_data["depth_path"]
    all_flow_paths = input_data["flow_path"]
    output_dir = input_data["output_dir"]
    frame_rate = input_data["frame_rate"]
    
    obj_dir = output_dir / "objects"
    mask_dir = output_dir / "masks"
    record_dir = output_dir / "records"
    gt_dir = output_dir / "gt"
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    data_num = input_data.get("data_num", "all")
    split_gt_json_files(input_data["gt_json_file"], gt_dir, data_num)
    
    if data_num != "all":
        data_num = int(data_num)
        all_video_paths = all_video_paths[:data_num]
        all_depth_paths = all_depth_paths[:data_num]
        all_frame_paths = all_frame_paths[:data_num]
        all_video_ids = all_video_ids[:data_num]
        all_flow_paths = all_flow_paths[:data_num]

    print(f"- Total Videos: {len(all_video_paths)}")
    for vid, v_path, f_path, d_path, flow_path in tqdm(zip(all_video_ids, all_video_paths, all_frame_paths, all_depth_paths, all_flow_paths), total=len(all_video_ids)):
        frame_paths = _video_to_frames(v_path, f_path)
        low_fps_frame_paths = _video_to_low_fps_frames(frame_rate, frame_paths)
        if input_data["use_gt"]:
            break 
        else:
            _frames_to_objects(vid, frame_rate, low_fps_frame_paths, obj_dir, od_model)
            _frames_to_masks(vid, frame_rate, low_fps_frame_paths, mask_dir, mask_model, mask_label_top_k)
            _frames_to_depths(frame_rate, low_fps_frame_paths, d_path, depth_model)
            _frames_to_flows(frame_rate, low_fps_frame_paths, flow_path, flow_model)
            _frames_to_records(vid, frame_rate, low_fps_frame_paths, d_path, flow_path, obj_dir, mask_dir, record_dir, packing_model)


    print("\n--------- Step 01 Done ---------------\n")

    
