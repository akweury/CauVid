"""
Test script for circle detection in CauVid synthetic videos

This script tests the circle detection capabilities and provides debugging information
to help optimize the detection parameters.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Add src to path to import pipeline modules
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import ObjectDetector, VideoFrameLoader
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_circle_detection():
    """Test circle detection on actual frames from observations."""
    
    TWO_PARTS_ROOT = "./two_parts"
    
    try:
        # Initialize components
        frame_loader = VideoFrameLoader(TWO_PARTS_ROOT)
        circle_detector = ObjectDetector(model_type="circle")
        
        # Get first observation
        observation_ids = frame_loader.get_observation_ids()
        if not observation_ids:
            logger.error("No observations found!")
            return
        
        first_obs = observation_ids[0]
        logger.info(f"Testing on observation: {first_obs}")
        
        # Load frames
        frames = frame_loader.load_frames(first_obs)
        if not frames:
            logger.error("No frames loaded!")
            return
        
        # Load metadata for ground truth
        metadata = frame_loader.load_metadata(first_obs)
        
        # Test on first few frames
        for frame_idx in range(min(3, len(frames))):
            logger.info(f"\n--- Testing Frame {frame_idx} ---")
            frame = frames[frame_idx]
            
            # Get ground truth
            gt_objects = frame_loader.get_ground_truth_objects(first_obs, frame_idx)
            logger.info(f"Ground truth objects: {len(gt_objects)}")
            for i, obj in enumerate(gt_objects):
                logger.info(f"  GT {i+1}: pos=({obj['x']:.1f}, {obj['y']:.1f}), side={obj['side']}, color={obj['color']}")
            
            # Detect circles
            detected_objects = circle_detector.detect_objects(frame, frame_idx)
            logger.info(f"Detected objects: {len(detected_objects)}")
            for i, obj in enumerate(detected_objects):
                logger.info(f"  Det {i+1}: {obj.label} at {obj.center} (conf: {obj.confidence:.2f})")
            
            # Save annotated frame for visual inspection
            save_annotated_frame(frame, detected_objects, gt_objects, 
                               f"debug_frame_{frame_idx}.png", frame_idx)
        
        logger.info("Circle detection test completed!")
        
    except Exception as e:
        logger.error(f"Error in circle detection test: {e}")
        raise


def save_annotated_frame(frame, detected_objects, gt_objects, filename, frame_idx):
    """Save frame with detection and ground truth annotations."""
    
    # Create a copy for annotation
    annotated = frame.copy()
    
    # Draw ground truth in green
    for obj in gt_objects:
        x, y = int(obj['x']), int(obj['y'])
        # Draw circle (assuming radius ~16 from metadata)
        cv2.circle(annotated, (x, y), 16, (0, 255, 0), 2)  # Green circle
        cv2.putText(annotated, "GT", (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw detections in red
    for obj in detected_objects:
        x, y = int(obj.center[0]), int(obj.center[1])
        # Calculate radius from bbox
        bbox_w = obj.bbox[2] - obj.bbox[0]
        bbox_h = obj.bbox[3] - obj.bbox[1]
        radius = int(min(bbox_w, bbox_h) / 2)
        
        cv2.circle(annotated, (x, y), radius, (255, 0, 0), 2)  # Red circle
        cv2.putText(annotated, f"Det {obj.confidence:.2f}", (x-15, y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Convert RGB to BGR for OpenCV saving
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, annotated_bgr)
    logger.info(f"Saved annotated frame: {filename}")


def tune_circle_detection_params():
    """Interactive parameter tuning for circle detection."""
    
    logger.info("Starting interactive parameter tuning...")
    
    TWO_PARTS_ROOT = "./two_parts"
    frame_loader = VideoFrameLoader(TWO_PARTS_ROOT)
    observation_ids = frame_loader.get_observation_ids()
    
    if not observation_ids:
        logger.error("No observations found!")
        return
    
    # Load a test frame
    frames = frame_loader.load_frames(observation_ids[0])
    test_frame = frames[0]
    gt_objects = frame_loader.get_ground_truth_objects(observation_ids[0], 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(test_frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Test different parameter combinations
    param_combinations = [
        {"dp": 1, "minDist": 20, "param1": 50, "param2": 30, "minR": 8, "maxR": 25},
        {"dp": 1, "minDist": 25, "param1": 40, "param2": 25, "minR": 10, "maxR": 30},
        {"dp": 1, "minDist": 30, "param1": 60, "param2": 35, "minR": 5, "maxR": 40},
        {"dp": 2, "minDist": 30, "param1": 50, "param2": 30, "minR": 8, "maxR": 25},
    ]
    
    logger.info(f"Ground truth has {len(gt_objects)} objects at positions:")
    for i, obj in enumerate(gt_objects):
        logger.info(f"  {i+1}: ({obj['x']:.1f}, {obj['y']:.1f})")
    
    best_params = None
    best_score = 0
    
    for i, params in enumerate(param_combinations):
        logger.info(f"\nTesting parameter set {i+1}: {params}")
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=params["dp"],
            minDist=params["minDist"],
            param1=params["param1"],
            param2=params["param2"],
            minRadius=params["minR"],
            maxRadius=params["maxR"]
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            logger.info(f"  Detected {len(circles)} circles:")
            
            # Calculate simple matching score
            matches = 0
            for circle in circles:
                x, y, r = circle
                logger.info(f"    Circle at ({x}, {y}) radius {r}")
                
                # Check if close to any ground truth
                for gt_obj in gt_objects:
                    gt_x, gt_y = gt_obj['x'], gt_obj['y']
                    distance = np.sqrt((x - gt_x)**2 + (y - gt_y)**2)
                    if distance < 20:  # Within 20 pixels
                        matches += 1
                        break
            
            score = matches / max(len(gt_objects), len(circles))  # Precision-like score
            logger.info(f"  Match score: {score:.2f} ({matches}/{len(gt_objects)} objects matched)")
            
            if score > best_score:
                best_score = score
                best_params = params
        else:
            logger.info("  No circles detected")
    
    if best_params:
        logger.info(f"\nBest parameters (score: {best_score:.2f}): {best_params}")
        logger.info("Consider updating the _detect_circles method with these parameters.")
    else:
        logger.info("\nNo good parameter combination found. May need to adjust approach.")


def create_synthetic_test_image():
    """Create a synthetic test image with known circles for debugging."""
    
    # Create a 224x224 image with some circles
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Add some colored circles
    circles_info = [
        {"center": (50, 50), "radius": 16, "color": (255, 0, 0)},    # Red
        {"center": (150, 80), "radius": 18, "color": (0, 255, 0)},   # Green
        {"center": (100, 150), "radius": 15, "color": (0, 0, 255)},  # Blue
        {"center": (180, 180), "radius": 20, "color": (255, 255, 0)} # Yellow
    ]
    
    for circle in circles_info:
        cv2.circle(img, circle["center"], circle["radius"], circle["color"], -1)
    
    # Add some noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Test detection
    detector = ObjectDetector(model_type="circle")
    detected = detector.detect_objects(img, 0)
    
    logger.info(f"Synthetic test: Created {len(circles_info)} circles, detected {len(detected)}")
    
    # Save test image
    cv2.imwrite("synthetic_test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return len(detected) == len(circles_info)


def main():
    """Main function to run various tests."""
    
    print("CauVid Circle Detection Testing")
    print("=" * 40)
    
    try:
        # Test 1: Synthetic image
        print("\n1. Testing on synthetic image...")
        synthetic_success = create_synthetic_test_image()
        print(f"Synthetic test {'PASSED' if synthetic_success else 'FAILED'}")
        
        # Test 2: Real observation frames
        print("\n2. Testing on observation frames...")
        test_circle_detection()
        
        # Test 3: Parameter tuning
        print("\n3. Parameter tuning...")
        tune_circle_detection_params()
        
        print("\nTesting completed! Check the generated debug images.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()