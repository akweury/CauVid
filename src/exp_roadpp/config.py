
import os
import cv2 as cv
from pathlib import Path
import shutil
from dataclasses import dataclass

# -------------- Settings --------------
ALLOW_MODEL_DOWNLOAD = True

# -------------- Path Settings --------------
root = Path(__file__).parents[0]
print(f"\n##### Root path: {root}\n")


exp_config_path = root/ "exp_config"
os.makedirs(exp_config_path, exist_ok=True)


# -------------- Pipeline Settings --------------


# -------------- Inputs --------------
def step_0_setup(args):
    # get the dataset phyiscal path
    # ml-pulsar
    if args.machine == "ml-pulsar":
        print("\n##### Running on ml-pulsar #####\n")
        args.device = "cuda:0"
        args.dataset_path = Path('/home/sha/mnt/remote/dgx-g/storage-01/CauVid_Data/roadpp')
        args.output_dir = root / 'output'/ 'roadpp' / args.exp
    # dgx
    elif args.machine == "dgx":
        print("\n##### Running on DGX #####\n")
        raise NotImplementedError("Not implemented yet...")
    
    # macbook pro
    elif args.machine == "macbook-pro":
        print("\n##### Running on MacBook Pro #####\n")
        args.device = "cpu"
        raise NotImplementedError("Not implemented yet...")

    # JST
    elif args.machine == "jst":
        print("\n##### Running on JST #####\n")
        args.device = "cuda:0"
        args.dataset_path = root / 'data' / 'roadpp'
        args.output_dir = root / 'output' / 'roadpp' / args.exp
        os.makedirs(root / 'output', exist_ok=True)
        os.makedirs(root / 'output' / 'roadpp', exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        raise ValueError(f"Add your own machine settings in config.py to setup device, output_dir, and dataset_path. "
                         f"Current machine: {args.machine}")


    # test the dataset path
    if os.path.exists(args.dataset_path):
        print(f"\n##### Dataset path exists: {args.dataset_path} #####\n")

    return args




def get_step_01_input(args):

    data_num = args.data_num
    if args.dataset == "bdd100k":
        video_dir = args.dataset_path / "videos"
        frame_dir = args.dataset_path / "frames"
        depth_dir = args.dataset_path / "depth_maps"
        flow_dir = args.dataset_path / "flows"
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(flow_dir, exist_ok=True)
            
        all_video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        all_frame_paths = [os.path.join(frame_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]
        all_depth_paths = [os.path.join(depth_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]
        all_video_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_video_paths]
        all_flow_paths = [os.path.join(flow_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]

        if data_num != "all":
            data_num = int(data_num)
            all_video_paths = all_video_paths[:data_num]
            all_depth_paths = all_depth_paths[:data_num]
            all_frame_paths = all_frame_paths[:data_num]
            all_video_ids = all_video_ids[:data_num]
            all_flow_paths = all_flow_paths[:data_num]

            output_dir = args.output_dir / "step01_output"
            os.makedirs(output_dir, exist_ok=True)
            input_data = {
                "allow_model_download": ALLOW_MODEL_DOWNLOAD,
                "output_dir": output_dir,
                "device": args.device,
                "video_path": all_video_paths,
                "frame_path": all_frame_paths,
                "depth_path": all_depth_paths,
                "flow_path": all_flow_paths,
                "video_ids": all_video_ids,
                # bdd100k dataset
                "bdd100k_frame_rate": args.bdd100k_frame_rate,
                # yolov8 world
                "od_model_path": root / args.driving_mini_od_model,
                "classes": args.driving_mini_obj_classes,
                "frame_rate":  args.bdd100k_frame_rate,
                "primary_confidence": args.primary_confidence,
                "candidate_confidence": args.candidate_confidence,
                "nms_iou": args.nms_iou,
                "inference_size": args.inference_size,
                # sam2
                "mask_model_path": root / args.sam2_model,
                "sam_prompt_candidates": args.sam_prompt_candidates,
                # semantic segmentation
                "semseg_enabled": getattr(args, "semseg_enabled", False),
                "semseg_model_path": getattr(
                    args,
                    "semseg_model_path",
                    "nvidia/segformer-b5-finetuned-ade-640-640",
                ),
                "semseg_target_labels": getattr(
                    args,
                    "semseg_target_labels",
                    ["sky", "road", "sidewalk", "terrain", "building", "vegetation", "wall", "fence"],
                ),
                "mask_label_top_k": getattr(args, "mask_label_top_k", 3),
                # Depth Anything v3
                "depth_model": args.depth_model,
                "depth_process_resolution": args.depth_process_resolution,
                # Flow model
                "flow_consistency_threshold_px": args.flow_consistency_threshold_px,
                
            }
    elif args.dataset == "roadpp":
        video_dir = args.dataset_path / "videos"
        test_video_dir = args.dataset_path / "test_videos"
        frame_dir = args.dataset_path / "frames"
        depth_dir = args.dataset_path / "depth_maps"
        flow_dir = args.dataset_path / "flows"
        gt_json_file = args.dataset_path / "road_waymo_trainval_v1.0.json"
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(flow_dir, exist_ok=True)

        all_video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        all_test_video_paths = [os.path.join(test_video_dir, f) for f in os.listdir(test_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        all_frame_paths = [os.path.join(frame_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]
        all_depth_paths = [os.path.join(depth_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]
        all_video_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_video_paths]
        all_test_video_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_test_video_paths]
        all_flow_paths = [os.path.join(flow_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]

        output_dir = args.output_dir / "step01_output"
        os.makedirs(output_dir, exist_ok=True)

        input_data = {
                        "dataset_path": args.dataset_path,
                        "use_gt":args.use_gt,
                        "gt_json_file": gt_json_file,
                        "allow_model_download": ALLOW_MODEL_DOWNLOAD,
                        "video_ids": all_video_ids,
                        "video_path": all_video_paths,
                        "frame_path": all_frame_paths,
                        "depth_path": all_depth_paths,
                        "flow_path": all_flow_paths,
                        "test_video_path": all_test_video_paths,
                        "test_video_ids": all_test_video_ids,
                        "output_dir": output_dir,
                        "frame_rate":  args.frame_rate,
                        "device": args.device,
                        "primary_confidence": args.primary_confidence,
                        "nms_iou": args.nms_iou,
                        "mask_label_top_k": getattr(args, "mask_label_top_k", 3),
                        "data_num": args.data_num,
                        # Flow model
                        "flow_consistency_threshold_px": args.flow_consistency_threshold_px,                        
        }

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    

    return input_data


def get_step_02_input(args):
    output_dir = args.output_dir / "step02_output"
    os.makedirs(output_dir, exist_ok=True)
    input_data = {
        'use_gt': args.use_gt,
        "output_dir": output_dir,
        "device": args.device,
        "step01_output_dir": args.output_dir / "step01_output",
        "mask_iou_th": args.mask_iou_th,
        "top_k": args.tracker_top_k,
        "window_size": args.tracker_window_size,
        "frame_rate": args.frame_rate,
    }
    
    return input_data



def get_step_03_input(args):
    output_dir = args.output_dir / "step03_output"
    os.makedirs(output_dir, exist_ok=True)        
    input_data = {
        "use_gt": args.use_gt,
        "dataset_path": args.dataset_path,
        "output_dir": output_dir,
        "step01_output_dir": args.output_dir / "step01_output",
        "step02_output_dir": args.output_dir / "step02_output",
        "device": args.device,
    }
    return input_data


def get_step_04_input(args):
    output_dir = args.output_dir / "step04_output"
    os.makedirs(output_dir, exist_ok=True)
    input_data = {
        "output_dir": output_dir,
        "step03_output_dir": args.output_dir / "step03_output",
        "device": args.device,
    }
    return input_data