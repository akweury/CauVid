"""
Video Processing Pipeline

Main video processing pipeline that orchestrates all components:
- Frame loading from observation directories
- Object detection using various algorithms
- Precision evaluation against ground truth
- Time-series matrix creation for temporal analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

from data_models import DetectedObject, ObjectTrack, FrameData, PrecisionMetrics
from frame_loader import VideoFrameLoader
from object_detector import create_detector
from precision_evaluator import PrecisionEvaluator
from time_series_matrix import TimeSeriesObjectMatrix

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Main video processing pipeline that orchestrates all components."""
    
    def __init__(self, two_parts_root: str, output_dir: str = "./pipeline_output",
                 detector_type: str = "circle", position_threshold: float = 20.0,
                 frame_height: int = 480, frame_width: int = 640):
        """
        Initialize the video processing pipeline.
        
        Args:
            two_parts_root: Path to the two_parts data directory
            output_dir: Directory to save pipeline outputs
            detector_type: Type of object detector ("circle", "yolo")
            position_threshold: Threshold for matching objects (pixels)
            frame_height: Expected frame height
            frame_width: Expected frame width
        """
        self.two_parts_root = two_parts_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.frame_loader = VideoFrameLoader(two_parts_root)
        self.detector = create_detector(detector_type, position_threshold=position_threshold)
        self.evaluator = PrecisionEvaluator(position_threshold=position_threshold)
        
        # Pipeline configuration
        self.frame_height = frame_height
        self.frame_width = frame_width
        logger.info(f"Initialized VideoPipeline with root: {two_parts_root}, detector: {detector_type}, position_threshold: {position_threshold}px")
    
    def process_observation(self, observation_id: str, include_ground_truth: bool = True,
                          extract_features: bool = False, analyze_bonds: bool = False) -> Dict[str, Any]:
        """
        Process a complete observation through the pipeline.
        
        Args:
            observation_id: ID of the observation to process
            include_ground_truth: Whether to load and evaluate against ground truth
            extract_features: Whether to extract advanced features (requires separate modules)
            analyze_bonds: Whether to perform bond analysis (requires separate modules)
            
        Returns:
            Dictionary containing all processing results
        """
        logger.info(f"Processing observation: {observation_id}")
        
        # Load frames
        frames_data = self.frame_loader.load_frames(observation_id)
        
        # Load ground truth if requested
        ground_truth = {}
        if include_ground_truth:
            ground_truth = self.frame_loader.load_ground_truth(observation_id)
        
        # Process each frame
        all_detections = []
        precision_metrics = []
        
        for i, frame_data in enumerate(frames_data):
            # Load the actual frame image
            frame_img = self.frame_loader.load_single_frame(observation_id, i)
            
            logger.info(f"Processing frame {i+1}/{len(frames_data)}")
            
            # Detect objects
            detected_objects = self.detector.detect_objects(frame_img, i)
            frame_data.detected_objects = detected_objects
            all_detections.extend(detected_objects)
            
            # Evaluate precision if ground truth available
            if i in ground_truth:
                metrics = self.evaluator.evaluate_frame(detected_objects, ground_truth[i])
                frame_data.precision_metrics = metrics
                precision_metrics.append(metrics)
        
        # Create time-series matrix
        time_series_matrix = TimeSeriesObjectMatrix()
        for frame_data in frames_data:
            time_series_matrix.add_frame_data(frame_data.frame_idx, frame_data.detected_objects)
        
        # Calculate velocities
        time_series_matrix.calculate_velocities()
        
        # Export results
        base_filename = f"{observation_id}"
        
        # Export time-series matrix
        time_series_json_path = self.output_dir / f"{base_filename}_timeseries.json"
        time_series_matrix.export_to_json(str(time_series_json_path))
        
        time_series_numpy_path = self.output_dir / f"{base_filename}_timeseries.npz"
        time_series_matrix.export_to_numpy(str(time_series_numpy_path))
        
        # Calculate overall metrics
        overall_metrics = None
        if precision_metrics:
            overall_metrics = self.evaluator.evaluate_sequence(precision_metrics)
            
            logger.info("Overall Detection Performance:")
            logger.info(f"  Precision: {overall_metrics.precision:.3f}")
            logger.info(f"  Recall: {overall_metrics.recall:.3f}")
            logger.info(f"  F1-Score: {overall_metrics.f1_score:.3f}")
            logger.info(f"  Mean Position Error: {overall_metrics.mean_position_error:.2f} pixels")
            logger.info(f"  True Positives: {overall_metrics.true_positives}")
            logger.info(f"  False Positives: {overall_metrics.false_positives}")
            logger.info(f"  False Negatives: {overall_metrics.false_negatives}")
        
        # Get time-series statistics
        ts_stats = time_series_matrix.get_statistics()
        logger.info("Time-series Matrix Statistics:")
        logger.info(f"  Objects: {ts_stats['num_objects']}")
        if 'mean_velocity' in ts_stats:
            logger.info(f"  Mean velocity: ({ts_stats['mean_velocity'][0]:.2f}, {ts_stats['mean_velocity'][1]:.2f}) px/frame")
            logger.info(f"  Mean speed: {ts_stats['mean_speed']:.2f} px/frame")
            logger.info(f"  Max speed: {ts_stats['max_speed']:.2f} px/frame")
        
        logger.info(f"Completed processing observation {observation_id}")
        
        # Prepare results
        results = {
            'observation_id': observation_id,
            'frames_data': frames_data,
            'time_series_matrix': time_series_matrix,
            'overall_metrics': overall_metrics,
            'all_detections': all_detections,
            'ground_truth': ground_truth,
            'statistics': ts_stats
        }
        
        return results