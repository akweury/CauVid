
import os

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import config 
from src.exp_driving_videos import knowledge 
from src.exp_driving_videos import exp_driving_utils as utils
from src.exp_driving_videos.percept2matrix import percept2matrix
from src.exp_driving_videos.matrix2primitives import matrix2primitives


def build_segments_from_change_points(seg_range, num_frames):
    segments = np.zeros(num_frames, dtype=int) - 1  # Initialize segments array with -1 (indicating unassigned)
    current_segment_id = 0
    
    for seg in seg_range:
        start, end = seg
        segments[start:end] = current_segment_id
        current_segment_id += 1
    
    return segments
    
def visualize_segmentation(signals, segments, video_id, activity_percentile, min_segment_len):
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
    figure_path = config.get_output_path('driving_segmentation_visualization') / f"{video_id}_ap_{activity_percentile}_msl_{min_segment_len}.png"
    
    if figure_path.exists():
        print(f"Segmentation visualization figure for video {video_id} already exists, skipping visualization.")
        return
    signal_ego_w = signals['ego_w']
    signal_ego_vz = signals['ego_vz']
    signal_ego_vx = signals['ego_vx']
    # a list of random colors for up to 100 segments
    colors = plt.cm.get_cmap('tab20', 100).colors
    # shuffle the colors to make adjacent segments more distinguishable
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(colors)
    
    # the second row with four subplots:
    ego_w_img = utils.create_timeline_line_chart_img(signal_ego_w, 0, segments, colors,
                                            title=f'Ego Yaw Rate with Change Points - Frame {0}', width=20,height=3)

    ego_vx_img = utils.create_timeline_line_chart_img(signal_ego_vx, 0, segments, colors,
                                            title=f'Ego Velocity X with Segments - Frame {0}', width=20,height=3)
    ego_vz_img = utils.create_timeline_line_chart_img(signal_ego_vz, 0, segments, colors,
                                            title=f'Ego Velocity Z with Segments - Frame {0}', width=20,height=3)
    
    ego_w_sign_img = utils.create_timeline_line_chart_img(np.diff(signal_ego_w), 0, segments, colors,
                                        title=f'Ego Yaw Rate Sign - Frame {0}', width=20,height=3)
    ego_vx_sign_img = utils.create_timeline_line_chart_img(np.abs(np.diff(signal_ego_vx)), 0, segments, colors,
                                        title=f'Ego Velocity X Sign - Frame {0}', width=20,height=3)
    ego_vz_sign_img = utils.create_timeline_line_chart_img(np.abs(np.diff(signal_ego_vz)), 0, segments, colors,
                                        title=f'Ego Velocity Z Sign - Frame {0}', width=20,height=3)
    
    segment_img = utils.create_segment_feature_img(segments, 0, colors,
                                            title=f'{len(segments)} Segments with Features - Frame {0}', width=20, height=3)
    chart_imgs = [ego_w_img, ego_w_sign_img]
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
    
    
    
def visualize_segmentation_video(primitive_continued, signals, segments, video_id):
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
    
    # fig, axs = plt.subplots(4, 1, figsize=(20, 8))
    # axs[0].plot(signal_ego_w, label='Ego Yaw Rate (w)')
    # axs[0].set_title(f'Ego Yaw Rate with Change Points')
    # axs[0].legend() 
    # axs[1].plot(signal_ego_vx, label='Ego Velocity X (vx)')
    # axs[1].set_title(f'Ego Velocity X with Segments')
    # axs[1].legend()
    # axs[2].plot(signal_ego_vz, label='Ego Velocity Z (vz)')
    # axs[2].set_title(f'Ego Velocity Z with Segments')
    # axs[2].legend()
    # axs[3].set_title(f'Segments with Features')
    # # each segment is colored differently based on the label, 
    # # and the features are shown as text in the middle of the segment

    # for seg in segments:
    #     s, e, feat, label = seg['start'], seg['end'], seg['features'], seg['label']
    #     axs[0].axvspan(s, e, color=colors.get(label, "gray"), alpha=0.3)
    #     axs[1].axvspan(s, e, color=colors.get(label, "gray"), alpha=0.3)
    #     axs[2].axvspan(s, e, color=colors.get(label, "gray"), alpha=0.3)
    #     axs[3].text((s+e)/2, 0.5, f"{label}", ha='center', va='center', fontsize=12)
    #     axs[3].axvspan(s, e, color=colors.get(label, "gray"), alpha=0.3)
    # plt.tight_layout()
    # save_path = config.get_output_path('driving_segmentation_visualization') / f"{video_id}_segmentation.png"
    # plt.savefig(save_path)
    # plt.close()
    # print(f"Segmentation visualization saved to {save_path}")
    

def preprocess_signal(signal):    
    # normalization and smoothing
    signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    signal_smoothed = np.convolve(signal_normalized, np.ones(5)/5, mode='same')
    return signal_smoothed



def segment_time_series(signal, activity_percentile, min_segment_len):
    """
    primitive_continued: 2d np array: rows are time stpes, columns are objects
    Each cell is a dict containing the primitive information for one object at that time step
    
    Output: 1d list with size of the number of frames, each element is the id of the segment that the frame belongs to. 
    The segments are defined by the changes in the primitives of the objects.

    """        
    # detect the change points in the ego motion to define the segments
    # This function should return the indices of the change points based on the smoothed ego rotation
    # segs = knowledge.detect_change_points(signal,
                                        # activity_percentile=activity_percentile, 
                                        # min_segment_len=min_segment_len)  
    
    segs, smoothed_signed_signal, signed_signal = signal2segments(signal)
    seg_labels = build_segments_from_change_points(segs, signal.shape[0])    
    
    # visualize the segmentation process
    # visualize_segmentation(signal, signal, segs, seg_labels, video_id, seg_type)
    
    cuts = [cut[0] for cut in segs] + [segs[-1][1]]  # Get the start index of each segment and add the end index of the last segment as cuts
    res = {
        "seg_labels": seg_labels,
        "signal": signal,
        "cuts": cuts
    }
    return res


def extract_features_from_segments(segments, signals):
    
    features = []  # This should be a list of feature vectors corresponding to each segment
    for (s,e) in segments:
        
        feat = {
            "yaw_sum": signals['ego_w'][s:e].sum(),
            "yaw_range": signals['ego_w'][s:e].max() - signals['ego_w'][s:e].min(),
            "vz_mean": signals['ego_vz'][s:e].mean(),
            "vz_trend": (signals['ego_vz'][e-1] - signals['ego_vz'][s]) / (e - s + 1e-5),
            "vx_mean": signals['ego_vx'][s:e].mean(),
        }
        features.append(feat)
    
    
    
    return features


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



def build_atomic_segments(signals, video_id, activity_percentile, min_segment_len):

    # segment each signal to get candidate cuts
    w_seg = segment_time_series(signals['ego_w'], "w", video_id, activity_percentile=activity_percentile, min_segment_len=min_segment_len)
    z_seg = segment_time_series(signals['ego_vz'], "z", video_id, activity_percentile=activity_percentile, min_segment_len=min_segment_len)
    x_seg = segment_time_series(signals['ego_vx'], "x", video_id, activity_percentile=activity_percentile, min_segment_len=min_segment_len)    

    # all_cuts = sorted(list(set(w_seg["cuts"] + z_seg["cuts"] + x_seg["cuts"])))
    all_cuts = sorted(list(set(w_seg["cuts"])))   
    all_segs=[(all_cuts[i], all_cuts[i+1]) for i in range(len(all_cuts)-1)]
    
    merged = []
    for seg in all_segs:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        if seg[1] - seg[0] < min_segment_len:
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)

    return merged


def primitive_segmentation(primitive_continued, ego_motion, video_id, activity_percentile, min_segment_len):
    
    signals = {
        "ego_w": ego_motion[:, 2],
        "ego_vz": ego_motion[:, 1],
        "ego_vx": ego_motion[:, 0]
    }
    # preprocess (normalize and smooth) the signals before segmentation
    signals = preprocess_signal(signals["ego_w"])
    
    # build atomic segments based on the fused cuts
    all_segs = build_atomic_segments(signals,video_id,activity_percentile, min_segment_len)
    segment_features = extract_features_from_segments(all_segs, signals)
    segment_labels = classify_segments(segment_features,
                                       yaw_percentile=activity_percentile,
                                       vz_trend_percentile=activity_percentile)
    
    primitive_segmented =[]
    for (seg, feat, label) in zip(all_segs, segment_features, segment_labels):
        s,e = seg
        primitive_segmented.append({
            "start": s,
            "end": e,
            "features": feat,
            "label": label
        })
    
    
    # visualize the segmentation result and save the visualization
    visualize_segmentation(signals, primitive_segmented, video_id, activity_percentile, min_segment_len)
    
    # visualize_segmentation_video(primitive_continued, signals, primitive_segmented, video_id)
    return primitive_segmented

def smooth_sign_als(signed_signal):
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

def seg_and_merge(smoothed_signed_signal, signal):
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


def signal2segments(signal):
    signed_signal = np.sign(np.diff(signal))
    # if a sign flip only last within 4 frames, we consider it as noise and smooth it out
    smoothed_signed_signal = smooth_sign_als(signed_signal)
    # each continuous segment with the same sign is a segment.
    # merge the segments if the signal has a low variance within the segment (indicating it's mostly flat and the sign change is just noise)
    merged_segs = seg_and_merge(smoothed_signed_signal, signal)
    return merged_segs, smoothed_signed_signal, signed_signal
    
def visual_signal_segmentation(vid, ap, msl, signal, prim_segs):
    figure_path = config.get_output_path('driving_segmentation_visualization') / f"{vid}_ap_{ap}_msl_{msl}.png"
    # a list of random colors for up to 100 segments
    colors = plt.cm.get_cmap('tab20', 100).colors
    # shuffle the colors to make adjacent segments more distinguishable
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(colors)
    
    # the second row with four subplots:
    ego_w_img = utils.create_timeline_line_chart_img(signal, 0, prim_segs, colors,
                                            title=f'Ego Yaw Rate with Change Points - Frame {0}', width=20,height=3)
    
    
    
    segs, smoothed_signed_signal, signed_signal = signal2segments(signal)
    
    
    
    seg_signal = np.zeros_like(signal)
    for i, seg in enumerate(segs):
        s, e = seg
        seg_signal[s:e] = i  # Assign a unique value to each segment for visualization
    seg_signal[-1] = seg_signal[-2]  # Handle the last frame case
    ego_w_smoothed_img = utils.create_timeline_line_chart_img(smoothed_signed_signal, 0, prim_segs, colors,
                                        title=f'Ego Yaw Rate Smoothed Sign - Frame {0}', width=20,height=3)
    ego_w_seg_img = utils.create_timeline_line_chart_img(seg_signal, 0, prim_segs, colors,
                                        title=f'Ego Yaw Rate Segments - Frame {0}', width=20,height=3)    
    chart_imgs = [ego_w_img,ego_w_smoothed_img, ego_w_seg_img]
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
    
def signal2prim_segs(signal, ap, msl):
    signal = preprocess_signal(signal)
    seg = segment_time_series(signal, activity_percentile=ap, min_segment_len=msl)
    all_cuts = sorted(list(set(seg["cuts"])))   
    all_segs=[(all_cuts[i], all_cuts[i+1]) for i in range(len(all_cuts)-1)]
    
    prim_segs= []
    for seg in all_segs:
        s,e = seg
        prim_segs.append({
            "start": s,
            "end": e
        })
    return prim_segs

def main_test():
        # test the segmentation on one video
    vid = config.driving_demo_video_id
    matrix = percept2matrix(vid, save_matrices_flag=True)
    _, ego_motion = matrix2primitives(matrix, vid, visualize_ego=True, save_primitives=True)
    ego_w = ego_motion[:, 2]
    ego_vz = ego_motion[:, 1]
    ego_vx = ego_motion[:, 0]
    
    
    msl = 5
    for ap in range(1, 100, 1):
        ego_w_prim_segs = signal2prim_segs(ego_w, ap=ap, msl=msl)
        visual_signal_segmentation(vid, ap, msl, ego_w, ego_w_prim_segs)
        
        
        
if __name__ == "__main__":
    main_test()
    
