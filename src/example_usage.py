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
    """Run a simple example of the pipeline with advanced feature extraction."""
    
    # Configuration
    TWO_PARTS_ROOT = "./two_parts"  # Path to the two_parts directory
    OUTPUT_DIR = "./pipeline_output"
    
    logger.info("Starting CauVid Pipeline Example with Feature Extraction")
    
    try:
        # Initialize the pipeline with circle detector and bond analysis
        pipeline = VideoPipeline(
            TWO_PARTS_ROOT, 
            OUTPUT_DIR, 
            detector_type="circle", 
            position_threshold=20.0,
            bond_threshold=0.8,  # Bond similarity threshold
            frame_height=480,    # Frame dimensions for normalization
            frame_width=640
        )
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        logger.info("Pipeline Configuration:")
        print(json.dumps(summary, indent=2))
        
        # Get available observations
        observation_ids = pipeline.frame_loader.get_observation_ids()
        
        if not observation_ids:
            logger.error("No observations found! Please check the two_parts directory.")
            return
        
        # Process the first observation with full feature extraction
        example_observation = observation_ids[0]
        logger.info(f"Processing example observation: {example_observation}")
        
        # Process with feature extraction and bond analysis
        results = pipeline.process_observation(
            example_observation,
            include_ground_truth=True,
            extract_features=True,
            analyze_bonds=True
        )
        
        # Extract results
        matrix = results['object_centric_matrix']
        time_series_matrix = results['time_series_matrix']
        object_features = results.get('object_features', {})
        object_bonds = results.get('object_bonds', {})
        video_segments = results.get('video_segments', [])
        
        # Print processing results
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
        
        # Show feature extraction results
        if object_features:
            logger.info(f"\nFeature Extraction Results:")
            logger.info(f"  - Extracted features for {len(object_features)} objects")
            
            # Show sample features for first object
            first_obj_id = list(object_features.keys())[0]
            first_obj_features = object_features[first_obj_id]
            logger.info(f"  - Sample features for {first_obj_id} ({len(first_obj_features)} frames):")
            
            if first_obj_features:
                sample_frame = list(first_obj_features.keys())[0]
                features = first_obj_features[sample_frame]
                logger.info(f"    Frame {sample_frame}:")
                logger.info(f"      Mean velocity direction: {features.mean_velocity_direction:.3f} rad")
                logger.info(f"      Mean speed: {features.mean_speed:.3f}")
                logger.info(f"      Contact pattern: {features.contact_pattern:.3f}")
                logger.info(f"      Support pattern: {features.support_pattern:.3f}")
                logger.info(f"      Kinetic energy: {features.kinetic_energy:.3f}")
                logger.info(f"      Potential energy: {features.potential_energy:.3f}")
        
        # Show bond analysis results
        if object_bonds:
            logger.info(f"\nBond Analysis Results:")
            total_bonds = sum(len(bonds) for bonds in object_bonds.values())
            broken_bonds = sum(1 for bonds in object_bonds.values() for bond in bonds if not bond.bond_exists)
            bond_strength_avg = np.mean([bond.similarity for bonds in object_bonds.values() for bond in bonds])
            
            logger.info(f"  - Total bonds analyzed: {total_bonds}")
            logger.info(f"  - Bonds broken: {broken_bonds} ({broken_bonds/total_bonds*100:.1f}%)")
            logger.info(f"  - Average bond strength: {bond_strength_avg:.3f}")
            
            # Show sample bond analysis for first object
            first_obj_bonds = object_bonds[first_obj_id]
            if first_obj_bonds:
                logger.info(f"  - Sample bonds for {first_obj_id} ({len(first_obj_bonds)} transitions):")
                for i, bond in enumerate(first_obj_bonds[:3]):  # Show first 3 bonds
                    status = "BOND" if bond.bond_exists else "BREAK"
                    key_event = " (KEY EVENT)" if bond.key_event_detected else ""
                    logger.info(f"    {bond.frame_t}→{bond.frame_t1}: {bond.similarity:.3f} [{status}]{key_event}")
        
        # Show video segmentation results
        if video_segments:
            logger.info(f"\nVideo Segmentation Results:")
            logger.info(f"  - Total segments: {len(video_segments)}")
            
            total_frames = sum(seg.end_frame - seg.start_frame + 1 for seg in video_segments)
            logger.info(f"  - Total frames: {total_frames}")
            
            for i, segment in enumerate(video_segments):
                duration = segment.end_frame - segment.start_frame + 1
                logger.info(f"  - {segment.segment_id}: frames {segment.start_frame}-{segment.end_frame} "
                           f"({duration} frames, avg_bond={segment.avg_bond_strength:.3f}, "
                           f"reason={segment.break_reason})")
        
        # Show sample trajectory for first object
        if time_series_matrix.object_tracks:
            first_obj_id = list(time_series_matrix.object_tracks.keys())[0]
            trajectory = time_series_matrix.get_object_trajectory(first_obj_id)
            logger.info(f"\nSample trajectory for {first_obj_id}:")
            logger.info(f"  - Frames tracked: {len(trajectory['frame_indices'])}")
            logger.info(f"  - Start position: ({trajectory['position_x'][0]:.1f}, {trajectory['position_y'][0]:.1f})")
            if len(trajectory['position_x']) > 1:
                logger.info(f"  - End position: ({trajectory['position_x'][-1]:.1f}, {trajectory['position_y'][-1]:.1f})")
                displacement = np.sqrt((trajectory['position_x'][-1] - trajectory['position_x'][0])**2 + 
                                     (trajectory['position_y'][-1] - trajectory['position_y'][0])**2)
                logger.info(f"  - Total displacement: {displacement:.2f} px")
        
        # Log output files
        logger.info("\nOutput Files Created:")
        logger.info(f"  - Object-centric matrix: {OUTPUT_DIR}/{example_observation}_processed.json")
        logger.info(f"  - Time-series matrix: {OUTPUT_DIR}/{example_observation}_timeseries.json")
        logger.info(f"  - Time-series numpy: {OUTPUT_DIR}/{example_observation}_timeseries.npz")
        if object_features:
            logger.info(f"  - Feature vectors: {OUTPUT_DIR}/{example_observation}_features.json")
        if object_bonds:
            logger.info(f"  - Bond analysis: {OUTPUT_DIR}/{example_observation}_bonds.json")
        if video_segments:
            logger.info(f"  - Video segments: {OUTPUT_DIR}/{example_observation}_segments.json")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise


def run_batch_processing():
    """Example of processing multiple observations with feature analysis."""
    
    TWO_PARTS_ROOT = "./two_parts"
    OUTPUT_DIR = "./pipeline_output"
    
    logger.info("Starting Batch Processing with Feature Analysis")
    
    try:
        pipeline = VideoPipeline(
            TWO_PARTS_ROOT, 
            OUTPUT_DIR, 
            detector_type="circle", 
            position_threshold=20.0,
            bond_threshold=0.8
        )
        
        # Process first 3 observations (or all if less than 3)
        observation_ids = pipeline.frame_loader.get_observation_ids()[:3]
        
        results = {}
        for obs_id in observation_ids:
            logger.info(f"Processing {obs_id} with feature extraction...")
            result = pipeline.process_observation(
                obs_id,
                include_ground_truth=True,
                extract_features=True,
                analyze_bonds=True
            )
            
            # Extract summary statistics
            matrix = result['object_centric_matrix']
            time_series_matrix = result['time_series_matrix']
            object_features = result.get('object_features', {})
            video_segments = result.get('video_segments', [])
            
            results[obs_id] = {
                'num_frames': len(matrix.frames_data),
                'num_tracks': len(matrix.object_tracks),
                'num_ts_objects': len(time_series_matrix.object_tracks),
                'velocity_stats': time_series_matrix.get_velocity_statistics(),
                'num_features': len(object_features),
                'num_segments': len(video_segments),
                'avg_segment_length': np.mean([seg.end_frame - seg.start_frame + 1 for seg in video_segments]) if video_segments else 0
            }
        
        logger.info("\nBatch Processing Results:")
        for obs_id, stats in results.items():
            vel_stats = stats['velocity_stats']
            logger.info(f"  {obs_id}:")
            logger.info(f"    - {stats['num_frames']} frames, {stats['num_ts_objects']} tracked objects")
            logger.info(f"    - Mean speed: {vel_stats['mean_speed']:.2f} px/frame")
            logger.info(f"    - Features extracted: {stats['num_features']} objects")
            logger.info(f"    - Video segments: {stats['num_segments']} (avg length: {stats['avg_segment_length']:.1f} frames)")
            
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
        ('sklearn', 'scikit-learn'),
        ('scipy', 'scipy'),
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
    