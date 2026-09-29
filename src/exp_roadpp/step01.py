
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import json

from src.exp_roadpp import utils_data


def split_gt_json_files(gt_json_file, out_dir, data_num, all_video_ids, all_test_video_ids):
    all_train_gt_json_files = [out_dir/f"{f}_gt.json" for f in all_video_ids]
    label_file = out_dir / "label.json"
    # if any json files not exist, then load the gt json file
    if not any(os.path.exists(f) for f in all_train_gt_json_files) or not os.path.exists(label_file):
        json_data = utils_data.load_json(gt_json_file)
        utils_data.save_json({
            "all_input_labels": json_data["all_input_labels"],
            "all_av_action_labels": json_data["all_av_action_labels"],
            "av_action_labels": json_data["av_action_labels"],
            "agent_labels": json_data["agent_labels"],
            "action_labels": json_data["action_labels"],
            "loc_labels": json_data["loc_labels"],
            "duplex_labels": json_data["duplex_labels"],
            "triplet_labels": json_data["triplet_labels"],
            "old_loc_labels": json_data["old_loc_labels"],
            "label_types": json_data["label_types"],
            "all_duplex_labels": json_data["all_duplex_labels"],
            "all_loc_labels": json_data["all_loc_labels"],
            "all_agent_labels": json_data["all_agent_labels"],
            "all_action_labels": json_data["all_action_labels"],
            "duplex_childs": json_data["duplex_childs"],
            "triplet_childs": json_data["triplet_childs"]
        }, label_file)
        for vid, data in json_data['db'].items():
            if "agent_tubes" not in data:
                continue
            if vid in all_test_video_ids:
                json_file_name = os.path.join(out_dir, f"{vid}_gt_test.json")
            else:
                json_file_name = os.path.join(out_dir, f"{vid}_gt.json")
            if not os.path.exists(json_file_name):
                print(f"Creating GT JSON file: {json_file_name}")
                utils_data.save_json({
                    "vid": vid,
                    "data": data
                    }, json_file_name)
        

    

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



def visual_annotations(track_dir, output_dir):
    # Implement the visualization logic for annotations
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    gt_labels = utils_data.load_json(track_dir / "label.json") 
    av_action_visual_file = output_dir / "av_action_visualization.png"
    agent_class_visual_file = output_dir / "agent_class_visualization.png"
    agent_action_visual_file = output_dir / "agent_action_visualization.png"
    agent_location_visual_file = output_dir / "agent_location_visualization.png"
    all_av_actions = []
    video_av_action_nums = []
    action_percentage = {}
    action_average_length = {}
    agent_class_counts = {}
    agent_action_counts = {}
    agent_location_counts = {}
    for track_file in all_track_files:
        track_data = utils_data.load_json(track_file)
        agent_tubes = track_data["data"]["agent_tubes"]
        agent_action_tubes = track_data["data"]["action_tubes"]
        agent_location_tubes = track_data["data"]["loc_tubes"]
        for agent_id, agent_data in agent_tubes.items():
            class_id = agent_data["label_id"]
            if class_id not in agent_class_counts:
                agent_class_counts[class_id] = 0
            agent_class_counts[class_id] += 1

        for agent_id, location_data in agent_location_tubes.items():
            location_id = location_data["label_id"]
            if location_id not in agent_location_counts:
                agent_location_counts[location_id] = 0
            agent_location_counts[location_id] += 1

        for agent_id, action_data in agent_action_tubes.items():
            action_id = action_data["label_id"]
            if action_id not in agent_action_counts:
                agent_action_counts[action_id] = 0
            agent_action_counts[action_id] += 1

        av_action_tubes = track_data["data"]["av_action_tubes"]
        video_av_action_nums.append(len(av_action_tubes))
        for data in av_action_tubes.values():
            all_av_actions.append(
                {"label_id": data["label_id"],
                 "length": len(data["frames"])}
            )
    average_av_action_num = sum(video_av_action_nums) / len(video_av_action_nums) if video_av_action_nums else 0

    for item in all_av_actions:
        label_id = item["label_id"]
        length = item["length"]
        if label_id not in action_percentage:
            action_percentage[label_id] = 0
            action_average_length[label_id] = []
        action_percentage[label_id] += 1
        action_average_length[label_id].append(length)
    for label_id in action_percentage:
        action_percentage[label_id] = action_percentage[label_id] / len(video_av_action_nums) if video_av_action_nums else 0
        action_average_length[label_id] = sum(action_average_length[label_id]) / len(action_average_length[label_id]) if action_average_length[label_id] else 0

    # visual agent class counts, 
    if agent_class_counts:
        import matplotlib.pyplot as plt
        class_label_ids = sorted(list(agent_class_counts.keys()))
        class_labels = [gt_labels["agent_labels"][int(label)] for label in class_label_ids]
        counts = [agent_class_counts[label] for label in class_label_ids]

        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, counts, color='tab:blue', alpha=0.6)
        plt.xlabel('Agent Class')
        plt.ylabel('Counts')
        plt.title('Agent Class Counts')
        plt.savefig(agent_class_visual_file)
        plt.close()


    # visual agent action counts
    if agent_action_counts:
        import matplotlib.pyplot as plt
        action_label_ids = sorted(list(agent_action_counts.keys()))
        action_labels = [gt_labels["action_labels"][int(label)] for label in action_label_ids]
        counts = [agent_action_counts[label] for label in action_label_ids]

        plt.figure(figsize=(22,5))
        plt.bar(action_labels, counts, color='tab:green', alpha=0.6)
        plt.xlabel('Agent Action')
        plt.ylabel('Counts')
        plt.title('Agent Action Counts')
        plt.savefig(agent_action_visual_file)
        plt.close()
    # visual agent location counts
    if agent_location_counts:
        import matplotlib.pyplot as plt
        location_label_ids = sorted(list(agent_location_counts.keys()))
        location_labels = [gt_labels["loc_labels"][int(label)] for label in location_label_ids]
        counts = [agent_location_counts[label] for label in location_label_ids]

        plt.figure(figsize=(15, 10))
        plt.bar(location_labels, counts, color='tab:orange', alpha=0.6)
        plt.xticks(rotation=30)
        plt.xlabel('Agent Location')
        plt.ylabel('Counts')
        plt.title('Agent Location Counts')
        plt.savefig(agent_location_visual_file)
        plt.close()
    # visual action percentage and action average length
    if action_percentage:
        import matplotlib.pyplot as plt
        action_label_ids = sorted(list(action_percentage.keys()))
        action_labels = [gt_labels["av_action_labels"][int(label)] for label in action_label_ids]
        percentages = [action_percentage[label] for label in action_label_ids]
        avg_lengths = [action_average_length[label] for label in action_label_ids]

        fig, ax1 = plt.subplots(figsize=(10,5))

        color = 'tab:blue'
        ax1.set_xlabel('Action Label')
        ax1.set_ylabel('Action Percentage', color=color)
        ax1.bar(action_labels, percentages, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Action Length', color=color)
        ax2.plot(action_labels, avg_lengths, color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Action Percentage and Average Action Length')
        plt.savefig(av_action_visual_file)
        plt.close()

    
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
    dataset_path = input_data["dataset_path"]
    all_frame_paths = input_data["frame_path"]
    all_depth_paths = input_data["depth_path"]
    all_flow_paths = input_data["flow_path"]
    all_test_video_paths = input_data["test_video_path"]
    all_test_video_ids = input_data["test_video_ids"]
    output_dir = input_data["output_dir"]
    frame_rate = input_data["frame_rate"]
    
    obj_dir = output_dir / "objects"
    mask_dir = output_dir / "masks"
    record_dir = output_dir / "records"
    gt_dir = dataset_path / "gt"
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)



    data_num = input_data.get("data_num", "full")
    split_gt_json_files(input_data["gt_json_file"], gt_dir, data_num, all_video_ids,all_test_video_ids)
    
    if data_num != "full":
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

    # visual_annotations(gt_dir, output_dir)

    print("\n--------- Step 01 Done ---------------\n")


