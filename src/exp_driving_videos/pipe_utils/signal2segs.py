
import os

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import config 
from exp_driving_videos.pipe_utils import exp_driving_utils as utils
from exp_driving_videos.pipe_utils.percept2matrix import percept2matrix
from exp_driving_videos.pipe_utils.matrix2primitives import matrix2primitives


######################## Helper functions for signal segmentation ########################

def _smooth_sign_als(signed_signal):
    smoothed_signed_signal = signed_signal.copy()
    i = 0
    while i < len(smoothed_signed_signal):
        # Find the length of the current run of same values
        run_start = i
        current_value = smoothed_signed_signal[i]
        while i < len(smoothed_signed_signal) and smoothed_signed_signal[i] == current_value:
            i += 1
        run_end = i
        run_length = run_end - run_start
        
        # If the run is short (less than 4 frames) and surrounded by the same different value, smooth it
        if run_length < 4:
            prev_value = smoothed_signed_signal[run_start - 1] if run_start > 0 else None
            next_value = smoothed_signed_signal[run_end] if run_end < len(smoothed_signed_signal) else None
            
            if prev_value is not None and next_value is not None and prev_value == next_value:
                smoothed_signed_signal[run_start:run_end] = prev_value
     
     # if the first or last few frames are short runs of a different value, smooth them as well
    if len(smoothed_signed_signal) > 0:
        # Check the start of the signal
        run_start = 0
        current_value = smoothed_signed_signal[0]
        while run_start < len(smoothed_signed_signal) and smoothed_signed_signal[run_start] == current_value:
            run_start += 1
        if run_start < 4 and run_start < len(smoothed_signed_signal):
            next_value = smoothed_signed_signal[run_start]
            smoothed_signed_signal[:run_start] = next_value
        
        # Check the end of the signal
        run_end = len(smoothed_signed_signal) - 1
        current_value = smoothed_signed_signal[-1]
        while run_end >= 0 and smoothed_signed_signal[run_end] == current_value:
            run_end -= 1
        if len(smoothed_signed_signal) - 1 - run_end < 4 and run_end >= 0:
            prev_value = smoothed_signed_signal[run_end]
            smoothed_signed_signal[run_end + 1:] = prev_value           
    return smoothed_signed_signal

def _seg_and_merge(smoothed_signed_signal, signal):
    # the signal is splitted by the flip signs, save all the segment ranges in the segs
    segs = []
    current_sign = smoothed_signed_signal[0]
    seg_start = 0
    for i in range(1, len(smoothed_signed_signal)):
        if smoothed_signed_signal[i] != current_sign:
            segs.append((seg_start, i))
            seg_start = i
            current_sign = smoothed_signed_signal[i]
    segs.append((seg_start, len(smoothed_signed_signal)))   
    
    # merge the segments if the signal has a low variance within the segment
    merged_segs = []
    for seg in segs:
        s, e = seg
        if np.var(signal[s:e]) < 0.01:  # If the variance of the signal within the segment is low, merge it with the previous segment
            if merged_segs:
                merged_segs[-1] = (merged_segs[-1][0], e)
            else:
                merged_segs.append((s, e))
        else:
            merged_segs.append((s, e))
    
    return merged_segs
    
######################## Visualization functions for segmentation results #######################

def _vis_segs(vid, signal, segs, signal_name):
    figure_path = config.get_output_path('driving_segmentation_visualization') / f"segs_{vid}_{signal_name}.png"
    # a list of random colors for up to 100 segments
    colors = plt.cm.get_cmap('tab20', 100).colors
    # shuffle the colors to make adjacent segments more distinguishable
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(colors)
    
    # the second row with four subplots:
    ego_w_img = utils.create_timeline_line_chart_img(signal, 0, segs, colors, title=f'{signal_name}', width=20,height=3)
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

    print(f"Segmentation visualization figure saved to {figure_path}") 
    
    
def _vis_segs_video(primitive_continued, signals, segments, video_id):
    """
    Two rows:
    first row: the frame image 
    second row with four subplots:
    1. the raw ego motion signal (yaw rate) with change points highlighted,
    2. the ego vx signal with the segments colored differently,
    3. the ego vz signal with the segments colored differently,
    4. the segments colored differently with the extracted features (yaw_sum, yaw_range, vz_mean, vz_trend) 
    shown as text.
    
    """
    video_path = config.get_output_path('driving_segmentation_visualization') / f"{video_id}.mp4"
    
    if video_path.exists():
        print(f"Segmentation visualization video for video {video_id} already exists, skipping visualization.")
        return
    
    
    
    signal_ego_w = signals['ego_w']
    signal_ego_vz = signals['ego_vz']
    signal_ego_vx = signals['ego_vx']
    colors = {
        "turn_right": "red",
        "turn_left": "blue",
        "decelerate": "orange",
        "accelerate": "green",
        "straight": "gray"
    }
    
    
    visualized_frames = []
    for frame_index, frame_objs_primitive in tqdm(enumerate(primitive_continued), total=len(primitive_continued)):
        non_obj_index = next((i for i, obj in enumerate(frame_objs_primitive) if obj is not None), None)
        frame_img = utils.load_frame(frame_objs_primitive[non_obj_index]['frame'])
        
        # the second row with four subplots:
        ego_w_img = utils.create_timeline_line_chart_img(signal_ego_w, frame_index, segments, colors,
                                             title=f'Ego Yaw Rate with Change Points - Frame {frame_index}', width=10,height=3)
        ego_vx_img = utils.create_timeline_line_chart_img(signal_ego_vx, frame_index, segments, colors,
                                             title=f'Ego Velocity X with Segments - Frame {frame_index}', width=10,height=3)
        ego_vz_img = utils.create_timeline_line_chart_img(signal_ego_vz, frame_index, segments, colors,
                                             title=f'Ego Velocity Z with Segments - Frame {frame_index}', width=10,height=3)
        segment_img = utils.create_segment_feature_img(segments, frame_index, colors,
                                             title=f'Segments with Features - Frame {frame_index}', width=10, height=3)
        chart_imgs = [ego_w_img, ego_vx_img, ego_vz_img, segment_img]
        combined_chart_img = utils.combine_images_vertically(chart_imgs)
        
        combined_img = utils.combine_images_vertically([frame_img, combined_chart_img])
        visualized_frames.append(combined_img)
        
    utils.write_video(visualized_frames, video_path, fps=2)   
    print(f"Segmentation visualization video saved to {video_path}") 


####################### Main functions for signal segmentation and primitive segmentation #######################

def signal2segs(signal):
    """
    Input: a 1D signal (e.g., ego yaw rate over time)
    
    Output: a list of segments, where each segment is a tuple of (start_index, end_index) 
            indicating the range of frames that belong to the same segment.
    """
    # normalization
    signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    
    # window smoothing
    signal_smoothed = np.convolve(signal_normalized, np.ones(5)/5, mode='same')
    
    # signal flip sign 
    signed_signal = np.sign(np.diff(signal_smoothed))
    
    # if a sign flip only last within 4 frames, we consider it as noise and smooth it out
    smoothed_signed_signal = _smooth_sign_als(signed_signal)
    
    # each continuous segment with the same sign is a segment.
    # merge the segments if the signal has a low variance within the segment (indicating it's mostly flat and the sign change is just noise)
    merged_segs = _seg_and_merge(smoothed_signed_signal, signal)
    
    return merged_segs


def classify_segments(segment_features, yaw_percentile=70, vz_trend_percentile=70):
    # This should be a list of labels corresponding to each segment based on the extracted feature
    labels = []  
    
    yaw_ranges = [feat['yaw_range'] for feat in segment_features]
    vz_trends = [feat['vz_trend'] for feat in segment_features]
    
    yaw_thr = np.percentile(yaw_ranges, yaw_percentile)
    vz_trend_thr = np.percentile(vz_trends, vz_trend_percentile)
    
    for f in segment_features:
        if f["yaw_range"]>yaw_thr:
            if f["yaw_sum"]>0:
                labels.append("turn_left")
            else:
                labels.append("turn_right")
        elif f["vz_trend"] <-vz_trend_thr:        
            labels.append("decelerate")
        elif f["vz_trend"] > vz_trend_thr:
            labels.append("accelerate")
        else:
            # differentiate stop and straight based on the mean velocity
            if f["vz_mean"] < 0.05:
                labels.append("stop")
            else:
                labels.append("straight") 
    
    return labels

def _pre_steps(vid):
    matrix = percept2matrix(vid, save_matrices_flag=True)
    _, ego_motion = matrix2primitives(matrix, vid, visualize_ego=True, save_primitives=True)
    ego_w = ego_motion[:, 2]
    ego_vz = ego_motion[:, 1]
    ego_vx = ego_motion[:, 0]
    return ego_w 


def main_test():
    vid = config.driving_demo_video_id
    
    ego_w= _pre_steps(vid)   
    
    ego_w_segs = signal2segs(ego_w)
    
    _vis_segs(vid,ego_w, ego_w_segs, "ego_w")

        
if __name__ == "__main__":
    main_test()
    
