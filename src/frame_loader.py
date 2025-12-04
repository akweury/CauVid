"""
Video frame loading utilities.
"""

import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any
import logging

from data_models import FrameData, DetectedObject

logger = logging.getLogger(__name__)


class VideoFrameLoader:
    """Handles loading video frames and ground truth data from observation directories."""
    
    def __init__(self, two_parts_root: str):
        self.two_parts_root = Path(two_parts_root)
        self.observation_dir = self.two_parts_root / "observation"
    
    def get_observation_ids(self) -> List[str]:
        """Get list of available observation IDs."""
        if not self.observation_dir.exists():
            logger.warning(f"Observation directory not found: {self.observation_dir}")
            return []
        
        observation_ids = []
        for item in self.observation_dir.iterdir():
            if item.is_dir() and item.name.startswith("observation_"):
                observation_ids.append(item.name)
        
        return sorted(observation_ids)
    
    def load_frames(self, observation_id: str) -> List[FrameData]:
        """Load all frames for a given observation."""
        frames_dir = self.observation_dir / observation_id / "frames"
        
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # Get all frame files
        frame_files = []
        for ext in ['.jpg', '.png', '.jpeg']:
            frame_files.extend(frames_dir.glob(f"*{ext}"))
        
        if not frame_files:
            raise ValueError(f"No frame images found in {frames_dir}")
        
        # Sort frames by filename (assuming sequential naming)
        frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
        
        frames_data = []
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frame_data = FrameData(frame_idx=i, detected_objects=[])
                frames_data.append(frame_data)
                logger.debug(f"Loaded frame {i}: {frame_file.name}")
        
        logger.info(f"Loaded {len(frames_data)} frames for observation {observation_id}")
        return frames_data
    
    def load_ground_truth(self, observation_id: str) -> Dict[int, List[DetectedObject]]:
        """Load ground truth object annotations for an observation."""
        meta_file = self.observation_dir / observation_id / "meta.json"
        
        if not meta_file.exists():
            logger.warning(f"No ground truth found: {meta_file}")
            return {}
        
        try:
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            
            ground_truth = {}
            
            # Parse ground truth data based on the structure
            if 'frames' in meta_data:
                for frame_info in meta_data['frames']:
                    frame_idx = frame_info.get('frame_id', 0)
                    objects = []
                    
                    if 'objects' in frame_info:
                        for obj_info in frame_info['objects']:
                            obj = DetectedObject(
                                label=obj_info.get('label', 'unknown'),
                                confidence=1.0,  # Ground truth has perfect confidence
                                bbox=tuple(obj_info.get('bbox', [0, 0, 0, 0])),
                                center=tuple(obj_info.get('center', [0, 0])),
                                area=obj_info.get('area', 0),
                                frame_idx=frame_idx
                            )
                            objects.append(obj)
                    
                    ground_truth[frame_idx] = objects
            
            logger.info(f"Loaded ground truth for {len(ground_truth)} frames")
            return ground_truth
            
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return {}
    
    def get_frame_path(self, observation_id: str, frame_idx: int) -> Path:
        """Get the file path for a specific frame."""
        frames_dir = self.observation_dir / observation_id / "frames"
        
        # Try common naming patterns
        for ext in ['.jpg', '.png', '.jpeg']:
            frame_path = frames_dir / f"frame_{frame_idx:06d}{ext}"
            if frame_path.exists():
                return frame_path
            
            frame_path = frames_dir / f"{frame_idx:06d}{ext}"
            if frame_path.exists():
                return frame_path
        
        raise FileNotFoundError(f"Frame {frame_idx} not found in {frames_dir}")
    
    def load_single_frame(self, observation_id: str, frame_idx: int):
        """Load a single frame image."""
        frame_path = self.get_frame_path(observation_id, frame_idx)
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            raise ValueError(f"Could not load frame: {frame_path}")
        
        return frame