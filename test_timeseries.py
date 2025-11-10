"""
Test script for time-series object matrix functionality

This script demonstrates the time-series matrix features for tracking
object properties across frames.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import VideoPipeline, TimeSeriesObjectMatrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_time_series_matrix():
    """Test the time-series matrix functionality."""
    
    TWO_PARTS_ROOT = "./two_parts"
    OUTPUT_DIR = "./timeseries_test_output"
    
    logger.info("Testing time-series object matrix functionality")
    
    try:
        # Initialize pipeline
        pipeline = VideoPipeline(
            TWO_PARTS_ROOT, 
            OUTPUT_DIR, 
            detector_type="circle",
            position_threshold=20.0
        )
        
        # Get first observation
        observation_ids = pipeline.frame_loader.get_observation_ids()
        if not observation_ids:
            logger.error("No observations found!")
            return
        
        test_obs = observation_ids[0]
        logger.info(f"Testing on: {test_obs}")
        
        # Process observation to get both matrices
        matrix, time_series_matrix = pipeline.process_observation(test_obs, include_ground_truth=True)
        
        # Analyze time-series matrix
        logger.info("\n" + "="*60)
        logger.info("TIME-SERIES MATRIX ANALYSIS")
        logger.info("="*60)
        
        # Basic information
        logger.info(f"Matrix dimensions: {time_series_matrix.num_frames} frames × {len(time_series_matrix.object_tracks)} objects")
        logger.info(f"Properties tracked: {time_series_matrix.property_names}")
        
        # Velocity statistics
        vel_stats = time_series_matrix.get_velocity_statistics()
        logger.info(f"\nVELOCITY STATISTICS:")
        logger.info(f"  Mean velocity: ({vel_stats['mean_velocity_x']:.2f}, {vel_stats['mean_velocity_y']:.2f}) px/frame")
        logger.info(f"  Velocity std: ({vel_stats['std_velocity_x']:.2f}, {vel_stats['std_velocity_y']:.2f}) px/frame")
        logger.info(f"  Mean speed: {vel_stats['mean_speed']:.2f} px/frame")
        logger.info(f"  Max speed: {vel_stats['max_speed']:.2f} px/frame")
        
        # Object trajectories analysis
        logger.info(f"\nOBJECT TRAJECTORIES:")
        for obj_id in list(time_series_matrix.object_tracks.keys())[:3]:  # Show first 3
            track = time_series_matrix.object_tracks[obj_id]
            trajectory = time_series_matrix.get_object_trajectory(obj_id)
            
            if len(trajectory['frame_indices']) > 1:
                start_pos = (trajectory['position_x'][0], trajectory['position_y'][0])
                end_pos = (trajectory['position_x'][-1], trajectory['position_y'][-1])
                displacement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                
                logger.info(f"  {obj_id}:")
                logger.info(f"    Frames: {len(trajectory['frame_indices'])}")
                logger.info(f"    Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
                logger.info(f"    End: ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
                logger.info(f"    Total displacement: {displacement:.2f} px")
                logger.info(f"    Color: RGB{track.color}")
        
        # Test matrix access methods
        test_matrix_access(time_series_matrix)
        
        # Create visualizations
        visualize_time_series_data(time_series_matrix, test_obs)
        
        return time_series_matrix
        
    except Exception as e:
        logger.error(f"Error in time-series test: {e}")
        raise


def test_matrix_access(ts_matrix: TimeSeriesObjectMatrix):
    """Test different ways to access the time-series matrix data."""
    
    logger.info(f"\nMATRIX ACCESS METHODS:")
    
    # Test property matrix access
    position_x_matrix = ts_matrix.get_property_matrix('position_x')
    position_y_matrix = ts_matrix.get_property_matrix('position_y')
    velocity_x_matrix = ts_matrix.get_property_matrix('velocity_x')
    
    logger.info(f"Position X matrix shape: {position_x_matrix.shape}")
    logger.info(f"Position X range: {np.nanmin(position_x_matrix):.1f} to {np.nanmax(position_x_matrix):.1f}")
    
    # Test frame-specific access
    frame_0_data = ts_matrix.get_frame_data(0)
    logger.info(f"Frame 0 objects: {list(frame_0_data.keys())}")
    
    # Test numpy export
    numpy_data = ts_matrix.export_to_numpy()
    logger.info(f"Numpy export keys: {list(numpy_data.keys())}")
    logger.info(f"Object IDs: {numpy_data['object_ids']}")


def visualize_time_series_data(ts_matrix: TimeSeriesObjectMatrix, observation_id: str):
    """Create visualizations of the time-series data."""
    
    logger.info("Creating time-series visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Time-Series Object Matrix: {observation_id}', fontsize=16)
    
    # Get data for plotting
    object_ids = list(ts_matrix.object_tracks.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(object_ids)))
    
    # 1. Position trajectories
    ax1 = axes[0, 0]
    for i, obj_id in enumerate(object_ids):
        trajectory = ts_matrix.get_object_trajectory(obj_id)
        if len(trajectory['position_x']) > 1:
            ax1.plot(trajectory['position_x'], trajectory['position_y'], 
                    'o-', color=colors[i], alpha=0.7, linewidth=2, markersize=3, label=obj_id[:10])
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Object Trajectories')
    ax1.set_xlim(0, 224)
    ax1.set_ylim(224, 0)  # Flip Y for image coordinates
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # 2. Velocity over time
    ax2 = axes[0, 1]
    for i, obj_id in enumerate(object_ids):
        trajectory = ts_matrix.get_object_trajectory(obj_id)
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(trajectory['velocity_x'], trajectory['velocity_y'])]
        if len(speeds) > 1:
            ax2.plot(trajectory['frame_indices'], speeds, 
                    'o-', color=colors[i], alpha=0.7, linewidth=1, markersize=2)
    
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Speed (pixels/frame)')
    ax2.set_title('Object Speeds Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Position X over time
    ax3 = axes[0, 2]
    for i, obj_id in enumerate(object_ids):
        trajectory = ts_matrix.get_object_trajectory(obj_id)
        if len(trajectory['position_x']) > 1:
            ax3.plot(trajectory['frame_indices'], trajectory['position_x'], 
                    'o-', color=colors[i], alpha=0.7, linewidth=1, markersize=2)
    
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('X Position (pixels)')
    ax3.set_title('X Position Over Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Position Y over time
    ax4 = axes[1, 0]
    for i, obj_id in enumerate(object_ids):
        trajectory = ts_matrix.get_object_trajectory(obj_id)
        if len(trajectory['position_y']) > 1:
            ax4.plot(trajectory['frame_indices'], trajectory['position_y'], 
                    'o-', color=colors[i], alpha=0.7, linewidth=1, markersize=2)
    
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Y Position (pixels)')
    ax4.set_title('Y Position Over Time')
    ax4.grid(True, alpha=0.3)
    
    # 5. Velocity distribution
    ax5 = axes[1, 1]
    all_vx = []
    all_vy = []
    for obj_id in object_ids:
        trajectory = ts_matrix.get_object_trajectory(obj_id)
        all_vx.extend([vx for vx in trajectory['velocity_x'] if abs(vx) < 50])
        all_vy.extend([vy for vy in trajectory['velocity_y'] if abs(vy) < 50])
    
    if all_vx and all_vy:
        ax5.hist2d(all_vx, all_vy, bins=20, alpha=0.7)
        ax5.set_xlabel('Velocity X (pixels/frame)')
        ax5.set_ylabel('Velocity Y (pixels/frame)')
        ax5.set_title('Velocity Distribution')
    
    # 6. Object size over time
    ax6 = axes[1, 2]
    for i, obj_id in enumerate(object_ids):
        trajectory = ts_matrix.get_object_trajectory(obj_id)
        areas = trajectory['area']
        if len(areas) > 1:
            ax6.plot(trajectory['frame_indices'], areas, 
                    'o-', color=colors[i], alpha=0.7, linewidth=1, markersize=2)
    
    ax6.set_xlabel('Frame Index')
    ax6.set_ylabel('Area (pixels²)')
    ax6.set_title('Object Size Over Time')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"timeseries_analysis_{observation_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved time-series analysis to {output_file}")
    
    plt.show()


def test_numpy_matrix_operations():
    """Test numpy matrix operations on the time-series data."""
    
    logger.info("Testing numpy matrix operations...")
    
    TWO_PARTS_ROOT = "./two_parts"
    OUTPUT_DIR = "./timeseries_test_output"
    
    pipeline = VideoPipeline(TWO_PARTS_ROOT, OUTPUT_DIR, detector_type="circle")
    observation_ids = pipeline.frame_loader.get_observation_ids()
    test_obs = observation_ids[0]
    
    # Process and get time-series matrix
    matrix, ts_matrix = pipeline.process_observation(test_obs)
    
    # Export to numpy
    numpy_data = ts_matrix.export_to_numpy()
    
    logger.info(f"Numpy matrices:")
    for key, array in numpy_data.items():
        if isinstance(array, np.ndarray) and array.ndim == 2:
            logger.info(f"  {key}: {array.shape} (mean: {np.nanmean(array):.2f})")
        elif isinstance(array, np.ndarray):
            logger.info(f"  {key}: {array.shape}")
    
    # Example analysis: calculate average displacement per object
    pos_x = numpy_data['position_x']
    pos_y = numpy_data['position_y']
    
    displacements = []
    for obj_idx in range(pos_x.shape[1]):
        x_values = pos_x[:, obj_idx]
        y_values = pos_y[:, obj_idx]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_values) | np.isnan(y_values))
        if np.sum(valid_mask) > 1:
            x_valid = x_values[valid_mask]
            y_valid = y_values[valid_mask]
            
            # Calculate total displacement
            displacement = np.sqrt((x_valid[-1] - x_valid[0])**2 + (y_valid[-1] - y_valid[0])**2)
            displacements.append(displacement)
    
    logger.info(f"Object displacements: mean={np.mean(displacements):.2f}, std={np.std(displacements):.2f}")


def main():
    """Main function to run time-series matrix tests."""
    
    print("CauVid Time-Series Object Matrix Testing")
    print("=" * 50)
    
    try:
        # Test 1: Basic time-series matrix functionality
        print("\n1. Testing time-series matrix creation...")
        ts_matrix = test_time_series_matrix()
        
        # Test 2: Numpy matrix operations
        print("\n2. Testing numpy matrix operations...")
        test_numpy_matrix_operations()
        
        print("\nTime-series matrix testing completed!")
        print("Check the generated output files and visualizations.")
        
    except Exception as e:
        logger.error(f"Time-series testing failed: {e}")
        raise


if __name__ == "__main__":
    main()