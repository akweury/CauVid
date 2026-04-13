
import os

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm

import config 
from src.exp_driving_videos.pipe_utils import exp_driving_utils as utils
from src.exp_driving_videos.pipe_utils.percept2matrix import percept2matrix
from src.exp_driving_videos.pipe_utils.matrix2signal import matrix2signal
from src.exp_driving_videos.pipe_utils.signal2segs import signal2segs




######################## Visualization functions #######################

def _vis_prims(vid, signal, segs, signal_name):
    raise NotImplementedError("This visualization function is not implemented yet, please implement it based on your needs. You can refer to the _vis_segs function in signal2segs.py for an example of how to visualize the segmentation results.")
    figure_path = config.get_output_path('driving_seg_feat_vis') / f"segs_{vid}_{signal_name}.png"
    # a list of random colors for up to 100 segments
    colors = plt.cm.get_cmap('tab20', 100).colors
    # shuffle the colors to make adjacent segments more distinguishable
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(colors)
    
    
    ego_w_img = utils.create_timeline_line_chart_img(signal, 0, segs, colors, title=f'Input {signal_name}', width=20,height=3)
    chart_imgs = [ego_w_img]
    # save the visualization as an image
    combined_chart_img = utils.combine_images_vertically(chart_imgs)
    # save np array as an image    
    plt.figure(figsize=(30, 10))
    plt.imshow(combined_chart_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    print(f"Feature visualization figure saved to {figure_path}") 


######################## Pipeline functions #########################
def seg2prim(seg_signal):
    # extract features for the segment, 
    # here we simply use the mean and variance of the signal within the segment as features
    prim = {
        "mean": np.mean(seg_signal),
        "min": np.min(seg_signal),
        "max": np.max(seg_signal),
        "amplitude": np.max(seg_signal) - np.min(seg_signal),
        "var": np.var(seg_signal),
        "slope": (seg_signal[-1] - seg_signal[0]) / (len(seg_signal) + 1e-6),  # to avoid division by zero
        "trend": 1 if seg_signal[-1] > seg_signal[0] else -1 if seg_signal[-1] < seg_signal[0] else 0,
        "mean_abs_diff": np.mean(np.abs(np.diff(seg_signal))),
        "max_abs_diff": np.max(np.abs(np.diff(seg_signal))),
        "energy": np.sum(seg_signal ** 2)
    }
    # Return as a single-row DataFrame
    return pd.DataFrame([prim])


def seg2prims(sig_seg, segs):
    # extract features for each segment, 
    # here we simply use the mean and variance of the signal within the segment as features
    prims = []
    for seg in segs:
        s, e = seg
        seg_signal = signal[s:e]
        # determine the trend of the signal within the segment: increasing, decreasing, or stable
        trend_type = 1 if seg_signal[-1] > seg_signal[0] else -1 if seg_signal[-1] < seg_signal[0] else 0
        
        prims.append({
            "start": s,
            "end": e,
            "duration": e - s,
            "mean": np.mean(seg_signal),
            "min": np.min(seg_signal),
            "max": np.max(seg_signal),
            "amplitude": np.max(seg_signal) - np.min(seg_signal),
            "var": np.var(seg_signal),
            "slope": (seg_signal[-1] - seg_signal[0]) / (e - s + 1e-6),  # to avoid division by zero
            "trend": trend_type,
            "mean_abs_diff": np.mean(np.abs(np.diff(seg_signal))),
            "max_abs_diff": np.max(np.abs(np.diff(seg_signal))),
            "energy": np.sum(seg_signal ** 2)            
        })
    return prims

    
######################## Test functions #########################
 
def _pre_steps(vid):
    matrix = percept2matrix(vid, save_matrices_flag=True)
    _, ego_motion = matrix2signal(matrix, vid, visualize_ego=True, save_primitives=True)
    ego_w = ego_motion[:, 2]
    ego_vz = ego_motion[:, 1]
    ego_vx = ego_motion[:, 0]
    ego_w_segs = signal2segs(ego_w)
    ego_vx_segs = signal2segs(ego_vx)
    ego_vz_segs = signal2segs(ego_vz)
    return ego_motion,ego_w_segs, ego_vx_segs, ego_vz_segs

def main_test():
    vid = config.driving_demo_video_id
    # Step 0: Preprocessing steps to get the ego motion signals and their segments
    ego_motion, ego_w_segs, ego_vx_segs, ego_vz_segs = _pre_steps(vid)
    signal_ego_w = ego_motion[:, 2]
    signal_ego_vz = ego_motion[:, 1]
    signal_ego_vx = ego_motion[:, 0]
    # Step 1: based on the segmentation and the signal, extract features for each segment.
    ego_w_prims = seg2prims(signal_ego_w, ego_w_segs)
    ego_vx_prims = seg2prims(signal_ego_vx, ego_vx_segs)
    ego_vz_prims = seg2prims(signal_ego_vz, ego_vz_segs)
       
    # Visualize the segmentation results for ego_w
    _vis_prims(vid,signal_ego_w, ego_w_segs, "ego_w")
    # Visualize the segmentation results for ego_vx
    _vis_prims(vid, signal_ego_vx, ego_vx_segs, "ego_vx")
    # Visualize the segmentation results for ego_vz
    _vis_prims(vid, signal_ego_vz, ego_vz_segs, "ego_vz")
    

        
if __name__ == "__main__":
    main_test()






































