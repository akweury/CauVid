"""
Object detection utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from data_models import DetectedObject

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Base class for object detection."""
    
    def __init__(self, position_threshold: float = 20.0):
        self.position_threshold = position_threshold
    
    def detect_objects(self, frame: np.ndarray, frame_idx: int) -> List[DetectedObject]:
        """Detect objects in a frame. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement detect_objects method")


class CircleDetector(ObjectDetector):
    """Circle detection using OpenCV HoughCircles."""
    
    def __init__(self, position_threshold: float = 20.0):
        super().__init__(position_threshold)
        logger.info("Using OpenCV HoughCircles for circle detection")
    
    def detect_objects(self, frame: np.ndarray, frame_idx: int) -> List[DetectedObject]:
        """Detect circular objects using HoughCircles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply HoughCircles with parameters tuned for 7 circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=5,           # Very small to allow overlapped circles
            param1=30,           # Lower edge threshold for solid circles
            param2=20,           # Higher accumulator threshold for solid circles
            minRadius=10,        # Around expected radius of 15
            maxRadius=20         # Around expected radius of 15
        )
        
        detected_objects = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print("Detected circle number:", len(circles))
            
            for i, (x, y, r) in enumerate(circles):
                # Convert circle to bounding box
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(frame.shape[1], x + r), min(frame.shape[0], y + r)
                
                # Calculate area
                area = np.pi * r * r
                
                # Extract most common RGB color
                rgb_color = self._get_most_common_color(frame, x, y, r)
                
                obj = DetectedObject(
                    label="circle",
                    confidence=0.95,  # High confidence for circle detection
                    bbox=(x1, y1, x2, y2),
                    center=(float(x), float(y)),
                    area=float(area),
                    frame_idx=frame_idx,
                    rgb_color=rgb_color
                )
                
                detected_objects.append(obj)
        
        return detected_objects
    
    def _get_most_common_color(self, frame: np.ndarray, x: int, y: int, r: int) -> Tuple[int, int, int]:
        """Extract the most common RGB color within the circle."""
        # Create circular mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Extract pixels within the circle
        pixels = frame[mask > 0]
        
        if len(pixels) == 0:
            return (0, 0, 0)
        
        # Convert BGR to RGB for correct color representation
        pixels_rgb = pixels[:, [2, 1, 0]]  # BGR to RGB conversion
        
        # Find the most common color by binning colors and finding the mode
        # Bin colors to reduce noise (group similar colors)
        binned_pixels = (pixels_rgb // 32) * 32  # Bin into groups of 32
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(binned_pixels.reshape(-1, 3), axis=0, return_counts=True)
        
        # Get the most common color
        most_common_idx = np.argmax(counts)
        most_common_color = unique_colors[most_common_idx]
        
        # Ensure values are in 0-255 range and convert to int
        most_common_color = np.clip(most_common_color, 0, 255).astype(int)
        
        return tuple(most_common_color)


class YOLODetector(ObjectDetector):
    """YOLO-based object detection."""
    
    def __init__(self, model_path: str = "yolov8n.pt", position_threshold: float = 20.0):
        super().__init__(position_threshold)
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model: {self.model_path}")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray, frame_idx: int) -> List[DetectedObject]:
        """Detect objects using YOLO."""
        if self.model is None:
            logger.error("YOLO model not loaded")
            return []
        
        results = self.model(frame)
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # Calculate center and area
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    obj = DetectedObject(
                        label=class_name,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        center=(float(center_x), float(center_y)),
                        area=float(area),
                        frame_idx=frame_idx
                    )
                    
                    detected_objects.append(obj)
        
        return detected_objects


def create_detector(detector_type: str, **kwargs) -> ObjectDetector:
    """Factory function to create object detectors."""
    if detector_type.lower() == "circle":
        return CircleDetector(**kwargs)
    elif detector_type.lower() in ["yolo", "yolov8"]:
        return YOLODetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")