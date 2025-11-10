"""
Visualization script for CauVid pipeline results

This script creates visualizations of the object detection and tracking results
to help analyze the pipeline performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import VideoFrameLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_detection_results(result_file: str, observation_id: str):
    """Create visualizations from pipeline results."""
    
    # Load results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    frames = results['frames']
    tracks = results['tracks']
    metadata = results['metadata']
    
    logger.info(f"Loaded results: {metadata['num_frames']} frames, {metadata['num_tracks']} tracks")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'CauVid Pipeline Results: {observation_id}', fontsize=16)
    
    # 1. Objects per frame over time
    ax1 = axes[0, 0]
    frame_indices = []
    object_counts = []
    
    for frame_idx_str, frame_data in frames.items():
        frame_indices.append(int(frame_idx_str))
        object_counts.append(frame_data['num_objects'])
    
    ax1.plot(frame_indices, object_counts, 'b-o', markersize=3)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Number of Objects Detected')
    ax1.set_title('Objects Detected per Frame')
    ax1.grid(True, alpha=0.3)
    
    # 2. Object trajectory visualization
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(tracks))))
    
    for i, (track_id, track_data) in enumerate(list(tracks.items())[:10]):  # Show first 10 tracks
        xs = [point['center'][0] for point in track_data]
        ys = [point['center'][1] for point in track_data]
        
        if len(xs) > 1:  # Only plot if there are multiple points
            ax2.plot(xs, ys, 'o-', color=colors[i], alpha=0.7, linewidth=2, markersize=4)
            ax2.text(xs[0], ys[0], f'T{i+1}', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
    
    ax2.set_xlabel('X Position (pixels)')
    ax2.set_ylabel('Y Position (pixels)')
    ax2.set_title('Object Trajectories (First 10 Tracks)')
    ax2.set_xlim(0, 224)
    ax2.set_ylim(224, 0)  # Flip Y axis for image coordinates
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Detection confidence distribution
    ax3 = axes[1, 0]
    confidences = []
    
    for frame_data in frames.values():
        for obj in frame_data['objects']:
            confidences.append(obj['confidence'])
    
    ax3.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Detection Confidence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Detection Confidence Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Object label distribution
    ax4 = axes[1, 1]
    label_counts = {}
    
    for frame_data in frames.values():
        for obj in frame_data['objects']:
            label = obj['label']
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
    
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    bars = ax4.bar(labels, counts, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Object Labels')
    ax4.set_ylabel('Total Detections')
    ax4.set_title('Object Label Distribution')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"visualization_{observation_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    
    plt.show()


def compare_ground_truth(observation_id: str, result_file: str):
    """Compare detection results with ground truth."""
    
    # Load pipeline results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Load ground truth
    loader = VideoFrameLoader("./two_parts")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Ground Truth vs Detection Comparison: {observation_id}', fontsize=14)
    
    # Analyze first few frames
    frames_to_analyze = min(10, len(results['frames']))
    
    gt_positions = {'x': [], 'y': []}
    det_positions = {'x': [], 'y': []}
    position_errors = []
    
    for frame_idx in range(frames_to_analyze):
        # Get ground truth
        gt_objects = loader.get_ground_truth_objects(observation_id, frame_idx)
        
        # Get detections
        frame_key = str(frame_idx)
        if frame_key in results['frames']:
            detected_objects = results['frames'][frame_key]['objects']
            
            # Simple matching: nearest neighbor
            for gt_obj in gt_objects:
                gt_x, gt_y = gt_obj['x'], gt_obj['y']
                gt_positions['x'].append(gt_x)
                gt_positions['y'].append(gt_y)
                
                # Find closest detection
                min_dist = float('inf')
                closest_det = None
                
                for det_obj in detected_objects:
                    det_x, det_y = det_obj['center']
                    dist = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_det = det_obj
                
                if closest_det:
                    det_x, det_y = closest_det['center']
                    det_positions['x'].append(det_x)
                    det_positions['y'].append(det_y)
                    position_errors.append(min_dist)
    
    # Plot 1: Position comparison
    ax1 = axes[0]
    ax1.scatter(gt_positions['x'], gt_positions['y'], c='red', alpha=0.6, s=50, label='Ground Truth')
    ax1.scatter(det_positions['x'], det_positions['y'], c='blue', alpha=0.6, s=50, label='Detections')
    
    # Draw connections between matched points
    for i in range(min(len(gt_positions['x']), len(det_positions['x']))):
        ax1.plot([gt_positions['x'][i], det_positions['x'][i]], 
                [gt_positions['y'][i], det_positions['y'][i]], 
                'g--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Ground Truth vs Detections')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 224)
    ax1.set_ylim(224, 0)
    ax1.set_aspect('equal')
    
    # Plot 2: Position error distribution
    ax2 = axes[1]
    if position_errors:
        ax2.hist(position_errors, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(np.mean(position_errors), color='red', linestyle='--', 
                   label=f'Mean Error: {np.mean(position_errors):.2f}px')
        ax2.set_xlabel('Position Error (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Detection Position Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        logger.info(f"Mean position error: {np.mean(position_errors):.2f} pixels")
        logger.info(f"Max position error: {np.max(position_errors):.2f} pixels")
    
    plt.tight_layout()
    
    # Save comparison plot
    output_file = f"comparison_{observation_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_file}")
    
    plt.show()


def main():
    """Main visualization function."""
    
    # Configuration
    observation_id = "observation_000000_862365"
    result_file = f"./pipeline_output/{observation_id}_processed.json"
    
    if not Path(result_file).exists():
        logger.error(f"Result file not found: {result_file}")
        logger.info("Please run the pipeline first: python example_usage.py")
        return
    
    print("CauVid Pipeline Visualization")
    print("=" * 40)
    
    try:
        # Create general visualizations
        print("Creating pipeline result visualizations...")
        visualize_detection_results(result_file, observation_id)
        
        # Create ground truth comparison
        print("Creating ground truth comparison...")
        compare_ground_truth(observation_id, result_file)
        
        print("Visualization complete! Check the generated PNG files.")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()