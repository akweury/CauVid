"""
CauVid Video Processing Pipeline

This pipeline processes video frames from observations to detect objects and create
object-centric representations. The pipeline includes:
1. Frame loading from observation directories
2. Object detection using pretrained models
3. Object-centric matrix creation for temporal analysis
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]  # x, y center coordinates
    area: float
    frame_idx: int


@dataclass
class ObjectTrack:
    """Represents a tracked object across multiple frames."""
    object_id: str
    label: str
    color: Optional[Tuple[int, int, int]] = None
    frames_data: Dict[int, Dict[str, Any]] = None  # frame_idx -> properties
    
    def __post_init__(self):
        if self.frames_data is None:
            self.frames_data = {}


@dataclass
class PrecisionMetrics:
    """Contains precision metrics for a frame or overall evaluation."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_position_error: float
    matched_pairs: List[Tuple[DetectedObject, Dict]]  # (detection, ground_truth) pairs


@dataclass
class FrameData:
    """Contains all information for a single frame."""
    frame_idx: int
    image: np.ndarray
    detected_objects: List[DetectedObject]
    ground_truth_objects: Optional[List[Dict]] = None
    precision_metrics: Optional[PrecisionMetrics] = None


class VideoFrameLoader:
    """Handles loading and preprocessing of video frames from observation directories."""
    
    def __init__(self, two_parts_root: str):
        self.two_parts_root = Path(two_parts_root)
        self.observation_dir = self.two_parts_root / "observation"
        
    def get_observation_ids(self) -> List[str]:
        """Get list of all observation IDs."""
        observation_dirs = [d.name for d in self.observation_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("observation_")]
        return sorted(observation_dirs)
    
    def load_metadata(self, observation_id: str) -> Dict[str, Any]:
        """Load metadata for a specific observation."""
        meta_path = self.observation_dir / observation_id / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
            
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def load_frames(self, observation_id: str) -> List[np.ndarray]:
        """Load all frames for a specific observation."""
        frames_dir = self.observation_dir / observation_id / "frames"
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        frames = []
        frame_files = sorted([f for f in frames_dir.iterdir() 
                            if f.suffix == '.png'])
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                logger.warning(f"Could not load frame: {frame_file}")
        
        logger.info(f"Loaded {len(frames)} frames for {observation_id}")
        return frames
    
    def get_ground_truth_objects(self, observation_id: str, frame_idx: int) -> List[Dict]:
        """Extract ground truth object information for a specific frame."""
        metadata = self.load_metadata(observation_id)
        
        if 'physics_simulation' not in metadata or 'frames' not in metadata['physics_simulation']:
            return []
            
        simulation_frames = metadata['physics_simulation']['frames']
        if frame_idx >= len(simulation_frames):
            return []
            
        return simulation_frames[frame_idx]['objects']


class ObjectDetector:
    """Handles object detection using pretrained models or specialized detectors for synthetic objects."""
    
    def __init__(self, model_type: str = "circle"):
        """Initialize the object detector.
        
        Args:
            model_type: Type of model to use ('circle', 'yolo', 'rcnn', etc.)
                       'circle' is optimized for synthetic circular objects
        """
        self.model_type = model_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained object detection model."""
        if self.model_type == "circle":
            # No model loading needed for circle detection - uses OpenCV HoughCircles
            logger.info("Using OpenCV HoughCircles for circle detection")
            self.model = "opencv_circles"
        elif self.model_type == "yolo":
            try:
                import ultralytics
                from ultralytics import YOLO
                # Load YOLOv8 model - will download on first use
                self.model = YOLO('yolov8n.pt')  # nano version for speed
                logger.info("Loaded YOLOv8 model successfully")
            except ImportError:
                logger.warning("ultralytics not available, using circle detector")
                self.model = "opencv_circles"
                self.model_type = "circle"
        else:
            logger.warning(f"Model type {self.model_type} not implemented, using circle detector")
            self.model = "opencv_circles"
            self.model_type = "circle"
    
    def detect_objects(self, frame: np.ndarray, frame_idx: int, 
                      confidence_threshold: float = 0.5) -> List[DetectedObject]:
        """Detect objects in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            frame_idx: Index of the frame
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detected objects
        """
        if self.model_type == "circle":
            return self._detect_circles(frame, frame_idx, confidence_threshold)
        elif self.model_type == "yolo" and self.model is not None:
            return self._detect_yolo(frame, frame_idx, confidence_threshold)
        else:
            # Fallback to dummy detector
            return self._dummy_detect(frame, frame_idx)
    
    def _detect_circles(self, frame: np.ndarray, frame_idx: int, 
                       confidence_threshold: float = 0.5) -> List[DetectedObject]:
        """Detect circular objects using OpenCV HoughCircles.
        
        This method is specifically designed for synthetic circular objects.
        """
        try:
            # Convert to grayscale for circle detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,           # Inverse ratio of accumulator resolution
                minDist=30,     # Minimum distance between circle centers
                param1=50,      # Upper threshold for edge detection
                param2=30,      # Accumulator threshold for center detection
                minRadius=5,    # Minimum circle radius
                maxRadius=50    # Maximum circle radius (adjust based on your data)
            )
            
            detected_objects = []
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for i, (x, y, radius) in enumerate(circles):
                    # Calculate bounding box
                    x1 = max(0, x - radius)
                    y1 = max(0, y - radius) 
                    x2 = min(frame.shape[1], x + radius)
                    y2 = min(frame.shape[0], y + radius)
                    
                    # Calculate area
                    area = np.pi * radius * radius
                    
                    # Assign confidence based on circle quality
                    # For synthetic circles, we can use radius consistency as a proxy
                    confidence = min(0.95, max(0.5, (radius / 25.0)))  # Normalize by expected radius
                    
                    # Extract color information from the center region
                    color_region = frame[max(0, y-5):min(frame.shape[0], y+5), 
                                       max(0, x-5):min(frame.shape[1], x+5)]
                    if color_region.size > 0:
                        avg_color = np.mean(color_region, axis=(0, 1))
                        color_label = self._classify_color(avg_color)
                        label = f"circle_{color_label}"
                    else:
                        label = "circle"
                    
                    detected_obj = DetectedObject(
                        label=label,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        center=(float(x), float(y)),
                        area=area,
                        frame_idx=frame_idx
                    )
                    detected_objects.append(detected_obj)
                    
                logger.debug(f"Detected {len(detected_objects)} circles in frame {frame_idx}")
            else:
                logger.debug(f"No circles detected in frame {frame_idx}")
                
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in circle detection: {e}")
            return []
    
    def _classify_color(self, rgb_color: np.ndarray) -> str:
        """Classify RGB color into basic color categories."""
        r, g, b = rgb_color
        
        # Simple color classification based on dominant channel
        if r > g and r > b:
            if r > 200:
                return "red"
            else:
                return "dark_red"
        elif g > r and g > b:
            if g > 200:
                return "green"
            else:
                return "dark_green"
        elif b > r and b > g:
            if b > 200:
                return "blue"
            else:
                return "dark_blue"
        else:
            # Similar values - could be gray, white, or mixed
            avg = (r + g + b) / 3
            if avg > 200:
                return "light"
            elif avg > 100:
                return "gray"
            else:
                return "dark"
    
    def _detect_yolo(self, frame: np.ndarray, frame_idx: int, 
                    confidence_threshold: float = 0.5) -> List[DetectedObject]:
        """Detect objects using YOLO model."""
        try:
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        label = self.model.names[class_id]
                        
                        # Calculate center and area
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        detected_obj = DetectedObject(
                            label=label,
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=(center_x, center_y),
                            area=area,
                            frame_idx=frame_idx
                        )
                        detected_objects.append(detected_obj)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def _dummy_detect(self, frame: np.ndarray, frame_idx: int) -> List[DetectedObject]:
        """Dummy object detector for demonstration purposes."""
        height, width = frame.shape[:2]
        
        # Create some dummy detections
        dummy_objects = [
            DetectedObject(
                label="circle",
                confidence=0.9,
                bbox=(50, 50, 100, 100),
                center=(75, 75),
                area=2500,
                frame_idx=frame_idx
            ),
            DetectedObject(
                label="object",
                confidence=0.8,
                bbox=(150, 100, 200, 150),
                center=(175, 125),
                area=2500,
                frame_idx=frame_idx
            )
        ]
        
        return dummy_objects


class PrecisionEvaluator:
    """Evaluates detection precision against ground truth data."""
    
    def __init__(self, position_threshold: float = 20.0):
        """Initialize the precision evaluator.
        
        Args:
            position_threshold: Maximum distance in pixels for a detection to be considered a match
        """
        self.position_threshold = position_threshold
        
    def calculate_frame_precision(self, detected_objects: List[DetectedObject], 
                                ground_truth_objects: List[Dict]) -> PrecisionMetrics:
        """Calculate precision metrics for a single frame.
        
        Args:
            detected_objects: List of detected objects
            ground_truth_objects: List of ground truth objects from metadata
            
        Returns:
            PrecisionMetrics containing all evaluation metrics
        """
        if not ground_truth_objects:
            # No ground truth available
            return PrecisionMetrics(
                true_positives=0,
                false_positives=len(detected_objects),
                false_negatives=0,
                precision=0.0 if len(detected_objects) > 0 else 1.0,
                recall=1.0,  # No ground truth to miss
                f1_score=0.0,
                mean_position_error=0.0,
                matched_pairs=[]
            )
        
        # Convert ground truth to more convenient format
        gt_objects = []
        for gt in ground_truth_objects:
            gt_objects.append({
                'x': gt['x'],
                'y': gt['y'],
                'color': gt['color'],
                'side': gt['side']
            })
        
        # Find matches using Hungarian algorithm (simplified nearest neighbor for now)
        matched_pairs = []
        used_detections = set()
        used_gt = set()
        position_errors = []
        
        # For each ground truth object, find the closest detection
        for gt_idx, gt_obj in enumerate(gt_objects):
            best_detection = None
            best_det_idx = None
            best_distance = float('inf')
            
            for det_idx, det_obj in enumerate(detected_objects):
                if det_idx in used_detections:
                    continue
                    
                # Calculate distance
                distance = np.sqrt((gt_obj['x'] - det_obj.center[0])**2 + 
                                 (gt_obj['y'] - det_obj.center[1])**2)
                
                if distance < best_distance and distance <= self.position_threshold:
                    best_distance = distance
                    best_detection = det_obj
                    best_det_idx = det_idx
            
            # If we found a match
            if best_detection is not None:
                matched_pairs.append((best_detection, gt_obj))
                used_detections.add(best_det_idx)
                used_gt.add(gt_idx)
                position_errors.append(best_distance)
        
        # Calculate metrics
        true_positives = len(matched_pairs)
        false_positives = len(detected_objects) - true_positives
        false_negatives = len(gt_objects) - true_positives
        
        precision = true_positives / len(detected_objects) if len(detected_objects) > 0 else 0.0
        recall = true_positives / len(gt_objects) if len(gt_objects) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_position_error = np.mean(position_errors) if position_errors else 0.0
        
        return PrecisionMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_position_error=mean_position_error,
            matched_pairs=matched_pairs
        )
    
    def calculate_overall_metrics(self, frame_metrics: List[PrecisionMetrics]) -> PrecisionMetrics:
        """Calculate overall precision metrics across all frames.
        
        Args:
            frame_metrics: List of precision metrics for each frame
            
        Returns:
            Overall precision metrics
        """
        if not frame_metrics:
            return PrecisionMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, [])
        
        total_tp = sum(m.true_positives for m in frame_metrics)
        total_fp = sum(m.false_positives for m in frame_metrics)
        total_fn = sum(m.false_negatives for m in frame_metrics)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Calculate mean position error across all frames
        all_errors = []
        for m in frame_metrics:
            if m.mean_position_error > 0:
                all_errors.extend([m.mean_position_error] * m.true_positives)
        mean_error = np.mean(all_errors) if all_errors else 0.0
        
        return PrecisionMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            mean_position_error=mean_error,
            matched_pairs=[]  # Don't aggregate matched pairs
        )


class TimeSeriesObjectMatrix:
    """
    Time-series object matrix for tracking object properties across frames.
    
    Structure: [frame_id, object_id, properties]
    - frame_id: Index of the frame
    - object_id: Unique identifier for the object
    - properties: Dictionary containing tracking properties
    """
    
    def __init__(self):
        self.matrix = {}  # frame_id -> object_id -> properties
        self.object_tracks = {}  # object_id -> ObjectTrack
        self.num_frames = 0
        self.property_names = [
            'position_x', 'position_y', 'velocity_x', 'velocity_y',
            'width', 'height', 'area', 'confidence', 'color_r', 'color_g', 'color_b', 'label'
        ]
    
    def add_object_detection(self, frame_idx: int, detected_object: DetectedObject, 
                           ground_truth_obj: Optional[Dict] = None):
        """Add a detected object to the time-series matrix."""
        
        # Extract object properties
        x, y = detected_object.center
        bbox = detected_object.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Calculate velocity (if previous frame exists)
        velocity_x, velocity_y = 0.0, 0.0
        object_id = self._get_or_create_object_id(detected_object, frame_idx)
        
        if object_id in self.object_tracks and frame_idx > 0:
            prev_frame = frame_idx - 1
            if prev_frame in self.object_tracks[object_id].frames_data:
                prev_data = self.object_tracks[object_id].frames_data[prev_frame]
                prev_x, prev_y = prev_data['position_x'], prev_data['position_y']
                velocity_x = x - prev_x
                velocity_y = y - prev_y
        
        # Extract color information
        color_r, color_g, color_b = 128, 128, 128  # Default gray
        if ground_truth_obj and 'color' in ground_truth_obj:
            color_r, color_g, color_b = ground_truth_obj['color']
        elif hasattr(detected_object, 'color') and detected_object.color:
            color_r, color_g, color_b = detected_object.color
        
        # Create properties dictionary
        properties = {
            'position_x': float(x),
            'position_y': float(y),
            'velocity_x': float(velocity_x),
            'velocity_y': float(velocity_y),
            'width': float(width),
            'height': float(height),
            'area': float(detected_object.area),
            'confidence': float(detected_object.confidence),
            'color_r': int(color_r),
            'color_g': int(color_g),
            'color_b': int(color_b),
            'label': detected_object.label
        }
        
        # Add to matrix
        if frame_idx not in self.matrix:
            self.matrix[frame_idx] = {}
        self.matrix[frame_idx][object_id] = properties
        
        # Update object track
        if object_id not in self.object_tracks:
            self.object_tracks[object_id] = ObjectTrack(
                object_id=object_id,
                label=detected_object.label,
                color=(color_r, color_g, color_b)
            )
        
        self.object_tracks[object_id].frames_data[frame_idx] = properties
        self.num_frames = max(self.num_frames, frame_idx + 1)
    
    def _get_or_create_object_id(self, detected_object: DetectedObject, frame_idx: int) -> str:
        """Get or create a unique object ID using simple tracking."""
        
        # For now, use a simple approach: try to match with previous frame objects
        if frame_idx == 0:
            # First frame: create new IDs
            existing_ids = [oid for oid in self.object_tracks.keys() if oid.startswith(detected_object.label)]
            object_id = f"{detected_object.label}_{len(existing_ids):03d}"
            return object_id
        
        # Find closest object from previous frame with same label
        prev_frame = frame_idx - 1
        if prev_frame in self.matrix:
            min_distance = float('inf')
            best_object_id = None
            
            for obj_id, prev_properties in self.matrix[prev_frame].items():
                if self.object_tracks[obj_id].label == detected_object.label:
                    # Calculate distance
                    prev_x = prev_properties['position_x']
                    prev_y = prev_properties['position_y']
                    curr_x, curr_y = detected_object.center
                    
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    
                    if distance < min_distance and distance < 50:  # 50 pixel threshold
                        min_distance = distance
                        best_object_id = obj_id
            
            if best_object_id:
                return best_object_id
        
        # Create new object ID if no match found
        existing_ids = [oid for oid in self.object_tracks.keys() if oid.startswith(detected_object.label)]
        object_id = f"{detected_object.label}_{len(existing_ids):03d}"
        return object_id
    
    def get_object_trajectory(self, object_id: str) -> Dict[str, List]:
        """Get the complete trajectory of an object across all frames."""
        
        if object_id not in self.object_tracks:
            return {}
        
        trajectory = {prop: [] for prop in self.property_names}
        trajectory['frame_indices'] = []
        
        track = self.object_tracks[object_id]
        for frame_idx in sorted(track.frames_data.keys()):
            trajectory['frame_indices'].append(frame_idx)
            frame_data = track.frames_data[frame_idx]
            
            for prop in self.property_names:
                trajectory[prop].append(frame_data.get(prop, 0))
        
        return trajectory
    
    def get_frame_data(self, frame_idx: int) -> Dict[str, Dict]:
        """Get all object data for a specific frame."""
        return self.matrix.get(frame_idx, {})
    
    def get_property_matrix(self, property_name: str) -> np.ndarray:
        """Get a 2D matrix for a specific property across all frames and objects."""
        
        if property_name not in self.property_names:
            raise ValueError(f"Property {property_name} not found. Available: {self.property_names}")
        
        # Get all unique object IDs
        all_object_ids = list(self.object_tracks.keys())
        
        # Create matrix [frames x objects]
        matrix = np.full((self.num_frames, len(all_object_ids)), np.nan)
        
        for frame_idx in range(self.num_frames):
            if frame_idx in self.matrix:
                for obj_idx, obj_id in enumerate(all_object_ids):
                    if obj_id in self.matrix[frame_idx]:
                        value = self.matrix[frame_idx][obj_id].get(property_name, np.nan)
                        if value is not None:
                            matrix[frame_idx, obj_idx] = float(value) if property_name != 'label' else hash(value) % 1000
        
        return matrix
    
    def get_velocity_statistics(self) -> Dict[str, float]:
        """Calculate velocity statistics across all objects."""
        
        all_vx = []
        all_vy = []
        all_speeds = []
        
        for obj_id in self.object_tracks:
            trajectory = self.get_object_trajectory(obj_id)
            vx_values = [v for v in trajectory['velocity_x'] if abs(v) < 100]  # Filter outliers
            vy_values = [v for v in trajectory['velocity_y'] if abs(v) < 100]
            
            all_vx.extend(vx_values)
            all_vy.extend(vy_values)
            
            # Calculate speeds
            for vx, vy in zip(vx_values, vy_values):
                speed = np.sqrt(vx**2 + vy**2)
                all_speeds.append(speed)
        
        return {
            'mean_velocity_x': np.mean(all_vx) if all_vx else 0.0,
            'mean_velocity_y': np.mean(all_vy) if all_vy else 0.0,
            'std_velocity_x': np.std(all_vx) if all_vx else 0.0,
            'std_velocity_y': np.std(all_vy) if all_vy else 0.0,
            'mean_speed': np.mean(all_speeds) if all_speeds else 0.0,
            'max_speed': np.max(all_speeds) if all_speeds else 0.0,
            'num_objects': len(self.object_tracks)
        }
    
    def export_to_numpy(self) -> Dict[str, np.ndarray]:
        """Export the complete matrix as numpy arrays for each property."""
        
        export_data = {}
        
        # Export each property as a 2D numpy array [frames x objects]
        for prop in self.property_names:
            if prop != 'label':  # Skip label for numpy export
                export_data[prop] = self.get_property_matrix(prop)
        
        # Add metadata
        export_data['object_ids'] = np.array(list(self.object_tracks.keys()))
        export_data['frame_indices'] = np.arange(self.num_frames)
        
        return export_data
    
    def export_to_json(self, output_path: str):
        """Export the time-series matrix to JSON format."""
        
        export_data = {
            'time_series_matrix': {},
            'object_tracks': {},
            'statistics': self.get_velocity_statistics(),
            'metadata': {
                'num_frames': self.num_frames,
                'num_objects': len(self.object_tracks),
                'property_names': self.property_names,
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Export frame-by-frame matrix
        for frame_idx in sorted(self.matrix.keys()):
            export_data['time_series_matrix'][str(frame_idx)] = self.matrix[frame_idx]
        
        # Export object trajectories
        for obj_id in self.object_tracks:
            trajectory = self.get_object_trajectory(obj_id)
            export_data['object_tracks'][obj_id] = {
                'label': self.object_tracks[obj_id].label,
                'color': self.object_tracks[obj_id].color,
                'trajectory': trajectory
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported time-series matrix to {output_path}")
        
        # Log statistics
        stats = export_data['statistics']
        logger.info(f"Time-series Matrix Statistics:")
        logger.info(f"  Objects: {stats['num_objects']}")
        logger.info(f"  Mean velocity: ({stats['mean_velocity_x']:.2f}, {stats['mean_velocity_y']:.2f}) px/frame")
        logger.info(f"  Mean speed: {stats['mean_speed']:.2f} px/frame")
        logger.info(f"  Max speed: {stats['max_speed']:.2f} px/frame")


class ObjectCentricMatrix:
    """Stores and manages object-centric representations across frames."""
    
    def __init__(self):
        self.frames_data: Dict[int, FrameData] = {}
        self.object_tracks: Dict[str, List[DetectedObject]] = {}
        self.overall_metrics: Optional[PrecisionMetrics] = None
        
    def add_frame_data(self, frame_data: FrameData):
        """Add frame data to the matrix."""
        self.frames_data[frame_data.frame_idx] = frame_data
        
        # Simple object tracking based on spatial proximity
        self._update_object_tracks(frame_data)
    
    def create_time_series_matrix(self) -> TimeSeriesObjectMatrix:
        """Create a time-series object matrix from the frame data."""
        
        time_series_matrix = TimeSeriesObjectMatrix()
        
        # Process all frames in order
        for frame_idx in sorted(self.frames_data.keys()):
            frame_data = self.frames_data[frame_idx]
            
            # Get ground truth objects for this frame (for color and other properties)
            gt_objects_dict = {}
            if frame_data.ground_truth_objects:
                for gt_obj in frame_data.ground_truth_objects:
                    # Create a key based on position for matching
                    gt_key = (round(gt_obj['x']), round(gt_obj['y']))
                    gt_objects_dict[gt_key] = gt_obj
            
            # Add each detected object to the time-series matrix
            for detected_obj in frame_data.detected_objects:
                # Try to find matching ground truth object
                det_key = (round(detected_obj.center[0]), round(detected_obj.center[1]))
                matching_gt = None
                
                # Find closest ground truth object
                if gt_objects_dict:
                    min_dist = float('inf')
                    for gt_key, gt_obj in gt_objects_dict.items():
                        dist = np.sqrt((det_key[0] - gt_key[0])**2 + (det_key[1] - gt_key[1])**2)
                        if dist < min_dist and dist < 20:  # 20 pixel threshold
                            min_dist = dist
                            matching_gt = gt_obj
                
                # Add to time-series matrix
                time_series_matrix.add_object_detection(
                    frame_idx, detected_obj, matching_gt
                )
        
        logger.info(f"Created time-series matrix: {time_series_matrix.num_frames} frames, "
                   f"{len(time_series_matrix.object_tracks)} objects")
        
        return time_series_matrix
    
    def _update_object_tracks(self, frame_data: FrameData):
        """Update object tracks with new frame data."""
        for obj in frame_data.detected_objects:
            # For now, create simple tracks based on label and frame
            track_id = f"{obj.label}_{obj.frame_idx}"
            
            if track_id not in self.object_tracks:
                self.object_tracks[track_id] = []
            
            self.object_tracks[track_id].append(obj)
    
    def get_object_trajectory(self, track_id: str) -> List[Tuple[float, float]]:
        """Get the trajectory of an object across frames."""
        if track_id not in self.object_tracks:
            return []
        
        trajectory = []
        for obj in sorted(self.object_tracks[track_id], key=lambda x: x.frame_idx):
            trajectory.append(obj.center)
        
        return trajectory
    
    def get_frame_summary(self, frame_idx: int) -> Dict[str, Any]:
        """Get a summary of objects in a specific frame."""
        if frame_idx not in self.frames_data:
            return {}
        
        frame_data = self.frames_data[frame_idx]
        
        summary = {
            'frame_idx': frame_idx,
            'num_objects': len(frame_data.detected_objects),
            'objects': []
        }
        
        for obj in frame_data.detected_objects:
            obj_info = {
                'label': obj.label,
                'confidence': obj.confidence,
                'center': obj.center,
                'bbox': obj.bbox,
                'area': obj.area
            }
            summary['objects'].append(obj_info)
        
        return summary
    
    def export_to_json(self, output_path: str):
        """Export the object-centric matrix to JSON format."""
        export_data = {
            'frames': {},
            'tracks': {},
            'precision_metrics': {},
            'metadata': {
                'num_frames': len(self.frames_data),
                'num_tracks': len(self.object_tracks),
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Export frame summaries and precision metrics
        for frame_idx in sorted(self.frames_data.keys()):
            frame_summary = self.get_frame_summary(frame_idx)
            export_data['frames'][str(frame_idx)] = frame_summary
            
            # Add precision metrics if available
            frame_data = self.frames_data[frame_idx]
            if frame_data.precision_metrics:
                metrics = frame_data.precision_metrics
                export_data['precision_metrics'][str(frame_idx)] = {
                    'true_positives': metrics.true_positives,
                    'false_positives': metrics.false_positives,
                    'false_negatives': metrics.false_negatives,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'mean_position_error': metrics.mean_position_error,
                    'num_matched_pairs': len(metrics.matched_pairs)
                }
        
        # Export overall metrics if available
        if self.overall_metrics:
            export_data['overall_precision'] = {
                'true_positives': self.overall_metrics.true_positives,
                'false_positives': self.overall_metrics.false_positives,
                'false_negatives': self.overall_metrics.false_negatives,
                'precision': self.overall_metrics.precision,
                'recall': self.overall_metrics.recall,
                'f1_score': self.overall_metrics.f1_score,
                'mean_position_error': self.overall_metrics.mean_position_error
            }
        
        # Export tracks
        for track_id, objects in self.object_tracks.items():
            track_data = []
            for obj in sorted(objects, key=lambda x: x.frame_idx):
                track_data.append({
                    'frame_idx': obj.frame_idx,
                    'center': obj.center,
                    'bbox': obj.bbox,
                    'confidence': obj.confidence,
                    'area': obj.area
                })
            export_data['tracks'][track_id] = track_data
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported object-centric matrix to {output_path}")
        
        # Log overall precision metrics
        if self.overall_metrics:
            logger.info(f"Overall Precision: {self.overall_metrics.precision:.3f}")
            logger.info(f"Overall Recall: {self.overall_metrics.recall:.3f}")
            logger.info(f"Overall F1-Score: {self.overall_metrics.f1_score:.3f}")
            logger.info(f"Mean Position Error: {self.overall_metrics.mean_position_error:.2f} pixels")


class VideoPipeline:
    """Main pipeline that orchestrates the entire video processing workflow."""
    
    def __init__(self, two_parts_root: str, output_dir: str = "output", 
                 detector_type: str = "circle", position_threshold: float = 20.0):
        """Initialize the pipeline.
        
        Args:
            two_parts_root: Path to the two_parts directory
            output_dir: Directory to save pipeline outputs
            detector_type: Type of object detector to use ('circle' for synthetic circles, 'yolo' for general objects)
            position_threshold: Maximum distance in pixels for matching detections to ground truth
        """
        self.frame_loader = VideoFrameLoader(two_parts_root)
        self.object_detector = ObjectDetector(model_type=detector_type)
        self.precision_evaluator = PrecisionEvaluator(position_threshold=position_threshold)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized VideoPipeline with root: {two_parts_root}, detector: {detector_type}, threshold: {position_threshold}px")
    
    def process_observation(self, observation_id: str, 
                          include_ground_truth: bool = True) -> ObjectCentricMatrix:
        """Process a single observation through the complete pipeline.
        
        Args:
            observation_id: ID of the observation to process
            include_ground_truth: Whether to include ground truth data
            
        Returns:
            ObjectCentricMatrix containing the processed results
        """
        logger.info(f"Processing observation: {observation_id}")
        
        # Load frames
        frames = self.frame_loader.load_frames(observation_id)
        
        # Initialize object-centric matrix
        matrix = ObjectCentricMatrix()
        
        # Store frame-level precision metrics
        frame_precision_metrics = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            logger.info(f"Processing frame {frame_idx + 1}/{len(frames)}")
            
            # Detect objects
            detected_objects = self.object_detector.detect_objects(frame, frame_idx)
            
            # Get ground truth if requested
            ground_truth = None
            precision_metrics = None
            
            if include_ground_truth:
                try:
                    ground_truth = self.frame_loader.get_ground_truth_objects(
                        observation_id, frame_idx)
                    
                    # Calculate precision metrics
                    precision_metrics = self.precision_evaluator.calculate_frame_precision(
                        detected_objects, ground_truth)
                    frame_precision_metrics.append(precision_metrics)
                    
                    # Log frame-level metrics
                    logger.debug(f"Frame {frame_idx}: P={precision_metrics.precision:.3f}, "
                               f"R={precision_metrics.recall:.3f}, F1={precision_metrics.f1_score:.3f}, "
                               f"Pos_Err={precision_metrics.mean_position_error:.2f}px")
                               
                except Exception as e:
                    logger.warning(f"Could not load ground truth for frame {frame_idx}: {e}")
            
            # Create frame data
            frame_data = FrameData(
                frame_idx=frame_idx,
                image=frame,
                detected_objects=detected_objects,
                ground_truth_objects=ground_truth,
                precision_metrics=precision_metrics
            )
            
            # Add to matrix
            matrix.add_frame_data(frame_data)
        
        # Calculate overall precision metrics
        if frame_precision_metrics:
            overall_metrics = self.precision_evaluator.calculate_overall_metrics(frame_precision_metrics)
            matrix.overall_metrics = overall_metrics
            
            logger.info(f"Overall Detection Performance:")
            logger.info(f"  Precision: {overall_metrics.precision:.3f}")
            logger.info(f"  Recall: {overall_metrics.recall:.3f}")
            logger.info(f"  F1-Score: {overall_metrics.f1_score:.3f}")
            logger.info(f"  Mean Position Error: {overall_metrics.mean_position_error:.2f} pixels")
            logger.info(f"  True Positives: {overall_metrics.true_positives}")
            logger.info(f"  False Positives: {overall_metrics.false_positives}")
            logger.info(f"  False Negatives: {overall_metrics.false_negatives}")
        
        # Export results
        output_file = self.output_dir / f"{observation_id}_processed.json"
        matrix.export_to_json(str(output_file))
        
        # Create and export time-series matrix
        logger.info("Creating time-series object matrix...")
        time_series_matrix = matrix.create_time_series_matrix()
        
        # Export time-series matrix
        ts_output_file = self.output_dir / f"{observation_id}_timeseries.json"
        time_series_matrix.export_to_json(str(ts_output_file))
        
        # Export as numpy arrays for easy analysis
        numpy_output_file = self.output_dir / f"{observation_id}_timeseries.npz"
        numpy_data = time_series_matrix.export_to_numpy()
        np.savez(str(numpy_output_file), **numpy_data)
        logger.info(f"Exported time-series numpy arrays to {numpy_output_file}")
        
        logger.info(f"Completed processing {observation_id}")
        return matrix, time_series_matrix
    
    def process_all_observations(self) -> Dict[str, Tuple[ObjectCentricMatrix, TimeSeriesObjectMatrix]]:
        """Process all available observations."""
        observation_ids = self.frame_loader.get_observation_ids()
        results = {}
        
        logger.info(f"Found {len(observation_ids)} observations to process")
        
        for observation_id in observation_ids:
            try:
                matrix, time_series_matrix = self.process_observation(observation_id)
                results[observation_id] = (matrix, time_series_matrix)
            except Exception as e:
                logger.error(f"Failed to process {observation_id}: {e}")
        
        return results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline configuration and capabilities."""
        observation_ids = self.frame_loader.get_observation_ids()
        
        summary = {
            'pipeline_info': {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'components': {
                    'frame_loader': 'VideoFrameLoader',
                    'object_detector': f'ObjectDetector({self.object_detector.model_type})',
                    'matrix_storage': 'ObjectCentricMatrix'
                }
            },
            'data_info': {
                'total_observations': len(observation_ids),
                'observation_ids': observation_ids,
                'data_root': str(self.frame_loader.two_parts_root)
            },
            'output_info': {
                'output_directory': str(self.output_dir)
            }
        }
        
        return summary


def main():
    """Main function to run the pipeline."""
    # Configuration
    TWO_PARTS_ROOT = "../two_parts"  # Adjust path as needed
    OUTPUT_DIR = "pipeline_output"
    
    # Initialize pipeline
    pipeline = VideoPipeline(TWO_PARTS_ROOT, OUTPUT_DIR)
    
    # Print pipeline summary
    summary = pipeline.get_pipeline_summary()
    logger.info(f"Pipeline Summary: {json.dumps(summary, indent=2)}")
    
    # Process first observation as example
    observation_ids = pipeline.frame_loader.get_observation_ids()
    if observation_ids:
        first_observation = observation_ids[0]
        logger.info(f"Processing example observation: {first_observation}")
        
        matrix = pipeline.process_observation(first_observation)
        
        # Print some results
        logger.info(f"Processed {len(matrix.frames_data)} frames")
        logger.info(f"Found {len(matrix.object_tracks)} object tracks")
        
        # Show summary of first frame
        if 0 in matrix.frames_data:
            frame_summary = matrix.get_frame_summary(0)
            logger.info(f"Frame 0 summary: {json.dumps(frame_summary, indent=2)}")


if __name__ == "__main__":
    main()