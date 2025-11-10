"""
Example usage of the CauVid Video Processing Pipeline

This script demonstrates how to use the pipeline to process video observations
and extract object-centric representations.
"""

import sys
import os
from pathlib import Path

# Add src to path to import pipeline modules
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import VideoPipeline, ObjectDetector
import json
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_example():
    """Run a simple example of the pipeline."""
    
    # Configuration
    TWO_PARTS_ROOT = "./two_parts"  # Path to the two_parts directory
    OUTPUT_DIR = "./pipeline_output"
    
    logger.info("Starting CauVid Pipeline Example")
    
    try:
        # Initialize the pipeline with circle detector and precision evaluation
        pipeline = VideoPipeline(TWO_PARTS_ROOT, OUTPUT_DIR, detector_type="circle", position_threshold=20.0)
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        logger.info("Pipeline Configuration:")
        print(json.dumps(summary, indent=2))
        
        # Get available observations
        observation_ids = pipeline.frame_loader.get_observation_ids()
        
        if not observation_ids:
            logger.error("No observations found! Please check the two_parts directory.")
            return
        
        # Process the first observation as an example
        example_observation = observation_ids[0]
        logger.info(f"Processing example observation: {example_observation}")
        
        # Process the observation
        matrix, time_series_matrix = pipeline.process_observation(example_observation)
        
        # Print results including precision metrics
        logger.info("Processing Results:")
        logger.info(f"  - Processed {len(matrix.frames_data)} frames")
        logger.info(f"  - Detected {len(matrix.object_tracks)} object tracks")
        
        # Show precision metrics if available
        if matrix.overall_metrics:
            logger.info(f"  - Overall Precision: {matrix.overall_metrics.precision:.3f}")
            logger.info(f"  - Overall Recall: {matrix.overall_metrics.recall:.3f}")
            logger.info(f"  - Overall F1-Score: {matrix.overall_metrics.f1_score:.3f}")
            logger.info(f"  - Mean Position Error: {matrix.overall_metrics.mean_position_error:.2f} pixels")
        
        # Show time-series matrix information
        logger.info("\nTime-Series Matrix Results:")
        logger.info(f"  - Matrix dimensions: {time_series_matrix.num_frames} frames × {len(time_series_matrix.object_tracks)} objects")
        
        # Show velocity statistics
        vel_stats = time_series_matrix.get_velocity_statistics()
        logger.info(f"  - Mean velocity: ({vel_stats['mean_velocity_x']:.2f}, {vel_stats['mean_velocity_y']:.2f}) px/frame")
        logger.info(f"  - Mean speed: {vel_stats['mean_speed']:.2f} px/frame")
        logger.info(f"  - Max speed: {vel_stats['max_speed']:.2f} px/frame")
        
        # Show sample trajectory for first object
        if time_series_matrix.object_tracks:
            first_obj_id = list(time_series_matrix.object_tracks.keys())[0]
            trajectory = time_series_matrix.get_object_trajectory(first_obj_id)
            logger.info(f"\nSample trajectory for {first_obj_id}:")
            logger.info(f"  - Frames tracked: {len(trajectory['frame_indices'])}")
            logger.info(f"  - Start position: ({trajectory['position_x'][0]:.1f}, {trajectory['position_y'][0]:.1f})")
            if len(trajectory['position_x']) > 1:
                logger.info(f"  - End position: ({trajectory['position_x'][-1]:.1f}, {trajectory['position_y'][-1]:.1f})")
                logger.info(f"  - Total displacement: {np.sqrt((trajectory['position_x'][-1] - trajectory['position_x'][0])**2 + (trajectory['position_y'][-1] - trajectory['position_y'][0])**2):.2f} px")
        
        # Show detailed results for first few frames
        for frame_idx in range(min(3, len(matrix.frames_data))):
            frame_summary = matrix.get_frame_summary(frame_idx)
            logger.info(f"Frame {frame_idx} summary:")
            print(f"  Objects detected: {frame_summary['num_objects']}")
            for i, obj in enumerate(frame_summary['objects']):
                print(f"    {i+1}. {obj['label']} (conf: {obj['confidence']:.2f}) at {obj['center']}")
        
        # Show object tracks
        if matrix.object_tracks:
            logger.info("Object Tracks:")
            for track_id, track_objects in list(matrix.object_tracks.items())[:3]:  # Show first 3 tracks
                trajectory = matrix.get_object_trajectory(track_id)
                logger.info(f"  Track {track_id}: {len(track_objects)} detections")
        
        logger.info(f"Results saved to: {OUTPUT_DIR}/{example_observation}_processed.json")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise


def run_batch_processing():
    """Example of processing multiple observations."""
    
    TWO_PARTS_ROOT = "./two_parts"
    OUTPUT_DIR = "./pipeline_output"
    
    logger.info("Starting Batch Processing Example")
    
    try:
        pipeline = VideoPipeline(TWO_PARTS_ROOT, OUTPUT_DIR, detector_type="circle", position_threshold=20.0)
        
        # Process first 3 observations (or all if less than 3)
        observation_ids = pipeline.frame_loader.get_observation_ids()[:3]
        
        results = {}
        for obs_id in observation_ids:
            logger.info(f"Processing {obs_id}...")
            matrix, time_series_matrix = pipeline.process_observation(obs_id)
            results[obs_id] = {
                'num_frames': len(matrix.frames_data),
                'num_tracks': len(matrix.object_tracks),
                'num_ts_objects': len(time_series_matrix.object_tracks),
                'velocity_stats': time_series_matrix.get_velocity_statistics()
            }
        
        logger.info("Batch Processing Results:")
        for obs_id, stats in results.items():
            vel_stats = stats['velocity_stats']
            logger.info(f"  {obs_id}: {stats['num_frames']} frames, {stats['num_ts_objects']} tracked objects")
            logger.info(f"    Mean speed: {vel_stats['mean_speed']:.2f} px/frame")
            
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise


def check_dependencies():
    """Check if required dependencies are installed."""
    
    logger.info("Checking dependencies...")
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
    ]
    
    optional_packages = [
        ('ultralytics', 'ultralytics (for YOLO object detection)'),
        ('torch', 'torch (for deep learning models)'),
    ]
    
    missing_required = []
    missing_optional = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} is installed")
        except ImportError:
            logger.error(f"✗ {name} is missing")
            missing_required.append(name)
    
    for package, name in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} is installed")
        except ImportError:
            logger.warning(f"! {name} is missing (optional)")
            missing_optional.append(name)
    
    if missing_required:
        logger.error("Missing required packages. Install with:")
        logger.error("pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        logger.warning("Missing optional packages for enhanced functionality:")
        logger.warning("pip install " + " ".join(missing_optional))
    
    return True


if __name__ == "__main__":
    print("CauVid Video Processing Pipeline - Example Usage")    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
        
        
        
    run_example()
    # run_batch_processing()
    