
import os

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import config 
from src.exp_driving_videos import knowledge 
from src.exp_driving_videos import exp_driving_utils as utils
    
    
def build_segments_from_change_points(seg_range, num_frames):
    segments = np.zeros(num_frames, dtype=int) - 1  # Initialize segments array with -1 (indicating unassigned)
    current_segment_id = 0
    
    for seg in seg_range:
        start, end = seg
        segments[start:end] = current_segment_id
        current_segment_id += 1
    
    return segments
    

def visualize_segmentation(primitive_continued, signals, segments, video_id):
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
    

def preprocess_signals(signals):
    """Preprocess the input signals by normalizing and smoothing them.
    Args:
        signals (dict): A dictionary where keys are signal names and values are 1D numpy arrays representing the signals.

    Returns:
        dict: A dictionary with the same keys as the input, where each signal has been normalized and smoothed.
    """
    
    preprocessed_signals = {}
    for key, signal in signals.items():
        # normalization and smoothing
        signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        signal_smoothed = np.convolve(signal_normalized, np.ones(5)/5, mode='same')
        preprocessed_signals[key] = signal_smoothed
    return preprocessed_signals



def segment_time_series(signal,seg_type, video_id, activity_percentile=96, min_segment_len=10):
    """
    primitive_continued: 2d np array: rows are time stpes, columns are objects
    Each cell is a dict containing the primitive information for one object at that time step
    
    Output: 1d list with size of the number of frames, each element is the id of the segment that the frame belongs to. 
    The segments are defined by the changes in the primitives of the objects.

    """        
    # detect the change points in the ego motion to define the segments
    # This function should return the indices of the change points based on the smoothed ego rotation
    segs = knowledge.detect_change_points(signal,
                                        activity_percentile=activity_percentile, 
                                        min_segment_len=min_segment_len)  
    
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



def build_atomic_segments(signals, video_id, activity_percentile=96, min_segment_len=10):

    # segment each signal to get candidate cuts
    w_seg = segment_time_series(signals['ego_w'], "w", video_id, activity_percentile=activity_percentile, min_segment_len=min_segment_len)
    z_seg = segment_time_series(signals['ego_vz'], "z", video_id, activity_percentile=activity_percentile, min_segment_len=min_segment_len)
    x_seg = segment_time_series(signals['ego_vx'], "x", video_id, activity_percentile=activity_percentile, min_segment_len=min_segment_len)    

    all_cuts = sorted(list(set(w_seg["cuts"] + z_seg["cuts"] + x_seg["cuts"])))   
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


def primitive_segmentation(primitive_continued, ego_motion, video_id, activity_percentile=96, min_segment_len=10):
    
    signals = {
        "ego_w": ego_motion[:, 2],
        "ego_vz": ego_motion[:, 1],
        "ego_vx": ego_motion[:, 0]
    }
    # preprocess (normalize and smooth) the signals before segmentation
    signals = preprocess_signals(signals)
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
    visualize_segmentation(primitive_continued, signals, primitive_segmented, video_id)
    return primitive_segmented

