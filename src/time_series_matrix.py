"""
Time Series Matrix Management

Handles creation and management of time-series object matrices from detection results.
Provides functionality for tracking objects across frames, calculating velocities,
and exporting results in various formats.
"""

import json
import numpy as np
from typing import Dict, List, Any
import logging

from data_models import DetectedObject, ObjectTrack

logger = logging.getLogger(__name__)


class TimeSeriesObjectMatrix:
    """Creates and manages time-series object matrices from detection results."""
    
    def __init__(self):
        self.matrix = {}  # frame_idx -> {object_id: properties}
        self.object_tracks = {}  # object_id -> ObjectTrack
        self.num_frames = 0
        self.property_names = []
    
    def add_frame_data(self, frame_idx: int, detected_objects: List[DetectedObject]):
        """Add detected objects for a frame to the matrix."""
        
        frame_data = {}
        
        for obj in detected_objects:
            # Generate object ID (simplified - in reality you'd want proper tracking)
            object_id = f"obj_{obj.frame_idx}_{hash(obj.center) % 1000:03d}"
            
            # Extract object properties
            properties = {
                'label': obj.label,
                'position_x': obj.center[0],
                'position_y': obj.center[1],
                'bbox_x1': obj.bbox[0],
                'bbox_y1': obj.bbox[1],
                'bbox_x2': obj.bbox[2],
                'bbox_y2': obj.bbox[3],
                'width': obj.bbox[2] - obj.bbox[0],
                'height': obj.bbox[3] - obj.bbox[1],
                'area': obj.area,
                'confidence': obj.confidence
            }
            
            frame_data[object_id] = properties
            
            # Update object track
            if object_id not in self.object_tracks:
                self.object_tracks[object_id] = ObjectTrack(
                    object_id=object_id,
                    label=obj.label,
                    frames_data={}
                )
            
            self.object_tracks[object_id].frames_data[frame_idx] = properties
        
        self.matrix[frame_idx] = frame_data
        self.num_frames = max(self.num_frames, frame_idx + 1)
        
        # Update property names
        if frame_data:
            sample_properties = next(iter(frame_data.values()))
            self.property_names = list(sample_properties.keys())
    
    def get_object_trajectory(self, object_id: str) -> List[Dict[str, Any]]:
        """Get the trajectory of a specific object across all frames."""
        
        if object_id not in self.object_tracks:
            return []
        
        trajectory = []
        track = self.object_tracks[object_id]
        
        for frame_idx in sorted(track.frames_data.keys()):
            trajectory.append({
                'frame_idx': frame_idx,
                **track.frames_data[frame_idx]
            })
        
        return trajectory
    
    def calculate_velocities(self):
        """Calculate velocity for each object between consecutive frames."""
        
        for object_id, track in self.object_tracks.items():
            frames = sorted(track.frames_data.keys())
            
            for i in range(1, len(frames)):
                prev_frame = frames[i-1]
                curr_frame = frames[i]
                
                prev_pos = (track.frames_data[prev_frame]['position_x'], 
                           track.frames_data[prev_frame]['position_y'])
                curr_pos = (track.frames_data[curr_frame]['position_x'],
                           track.frames_data[curr_frame]['position_y'])
                
                # Calculate velocity
                velocity_x = curr_pos[0] - prev_pos[0]
                velocity_y = curr_pos[1] - prev_pos[1]
                
                # Add velocity to current frame data
                track.frames_data[curr_frame]['velocity_x'] = velocity_x
                track.frames_data[curr_frame]['velocity_y'] = velocity_y
                
                # Update matrix
                if curr_frame in self.matrix and object_id in self.matrix[curr_frame]:
                    self.matrix[curr_frame][object_id]['velocity_x'] = velocity_x
                    self.matrix[curr_frame][object_id]['velocity_y'] = velocity_y
        
        # Update property names to include velocities
        if self.property_names and 'velocity_x' not in self.property_names:
            self.property_names.extend(['velocity_x', 'velocity_y'])
    
    def export_to_json(self, output_path: str):
        """Export the time-series matrix to JSON format."""
        
        export_data = {
            'num_frames': self.num_frames,
            'property_names': self.property_names,
            'object_tracks': {},
            'matrix': {}
        }
        
        # Export object tracks
        for object_id, track in self.object_tracks.items():
            export_data['object_tracks'][object_id] = {
                'object_id': track.object_id,
                'label': track.label,
                'frames_data': track.frames_data
            }
        
        # Export matrix (convert frame indices to strings for JSON)
        for frame_idx, frame_data in self.matrix.items():
            export_data['matrix'][str(frame_idx)] = frame_data
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported time-series matrix to {output_path}")
    
    def export_to_numpy(self, output_path: str):
        """Export the time-series matrix to NumPy format."""
        
        # Create dense matrix: [frames, objects, properties]
        unique_objects = list(self.object_tracks.keys())
        dense_matrix = np.full((self.num_frames, len(unique_objects), len(self.property_names)), np.nan)
        
        for frame_idx, frame_data in self.matrix.items():
            for obj_idx, object_id in enumerate(unique_objects):
                if object_id in frame_data:
                    for prop_idx, prop_name in enumerate(self.property_names):
                        if prop_name in frame_data[object_id]:
                            dense_matrix[frame_idx, obj_idx, prop_idx] = frame_data[object_id][prop_name]
        
        # Save as compressed numpy file
        np.savez_compressed(
            output_path,
            matrix=dense_matrix,
            object_ids=unique_objects,
            property_names=self.property_names,
            num_frames=self.num_frames
        )
        
        logger.info(f"Exported time-series numpy arrays to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the time-series matrix."""
        
        stats = {
            'num_frames': self.num_frames,
            'num_objects': len(self.object_tracks),
            'property_names': self.property_names,
            'objects_per_frame': {},
            'trajectory_lengths': {}
        }
        
        # Calculate objects per frame
        for frame_idx, frame_data in self.matrix.items():
            stats['objects_per_frame'][frame_idx] = len(frame_data)
        
        # Calculate trajectory lengths
        for object_id, track in self.object_tracks.items():
            stats['trajectory_lengths'][object_id] = len(track.frames_data)
        
        # Calculate velocity statistics if available
        if 'velocity_x' in self.property_names and 'velocity_y' in self.property_names:
            velocities = []
            speeds = []
            
            for track in self.object_tracks.values():
                for frame_data in track.frames_data.values():
                    if 'velocity_x' in frame_data and 'velocity_y' in frame_data:
                        vx, vy = frame_data['velocity_x'], frame_data['velocity_y']
                        velocities.append((vx, vy))
                        speeds.append(np.sqrt(vx*vx + vy*vy))
            
            if speeds:
                stats['mean_velocity'] = (np.mean([v[0] for v in velocities]), 
                                        np.mean([v[1] for v in velocities]))
                stats['mean_speed'] = np.mean(speeds)
                stats['max_speed'] = np.max(speeds)
        
        return stats