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
        
        # Object tracking state
        self.next_object_id = 0
        self.previous_objects = []  # Objects from previous frame for tracking
        self.tracking_threshold = 50.0  # Maximum distance for object matching
    
    def _calculate_distance(self, obj1_center, obj2_center):
        """Calculate Euclidean distance between two object centers."""
        return np.sqrt((obj1_center[0] - obj2_center[0])**2 + (obj1_center[1] - obj2_center[1])**2)
    
    def _find_matching_object(self, current_obj: DetectedObject):
        """Find the best matching object from the previous frame.
        
        Args:
            current_obj: Current detected object
            
        Returns:
            object_id if match found, None otherwise
        """
        if not self.previous_objects:
            return None
            
        best_match = None
        min_distance = float('inf')
        
        for prev_obj_id, prev_obj_data in self.previous_objects:
            # Check if same class/label
            if prev_obj_data['label'] != current_obj.label:
                continue
                
            # Calculate distance
            prev_center = (prev_obj_data['position_x'], prev_obj_data['position_y'])
            distance = self._calculate_distance(current_obj.center, prev_center)
            
            # Find closest match within threshold
            if distance < self.tracking_threshold and distance < min_distance:
                min_distance = distance
                best_match = prev_obj_id
                
        return best_match
    
    def _generate_new_object_id(self, obj_label: str):
        """Generate a new unique object ID."""
        object_id = f"{obj_label}_{self.next_object_id:03d}"
        self.next_object_id += 1
        return object_id

    def add_frame_data(self, frame_idx: int, detected_objects: List[DetectedObject]):
        """Add detected objects for a frame to the matrix with proper object tracking."""
        
        frame_data = {}
        current_frame_objects = []  # Store current frame objects for next frame tracking
        
        for obj in detected_objects:
            # Try to find matching object from previous frame
            object_id = self._find_matching_object(obj)
            
            # If no match found, create new object ID
            if object_id is None:
                object_id = self._generate_new_object_id(obj.label)
                logger.info(f"Created new object track: {object_id} at frame {frame_idx}")
            else:
                logger.debug(f"Matched existing object: {object_id} at frame {frame_idx}")
            
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
                'confidence': obj.confidence,
                "color_r": obj.rgb_color[0] if obj.rgb_color else None,
                "color_g": obj.rgb_color[1] if obj.rgb_color else None,
                "color_b": obj.rgb_color[2] if obj.rgb_color else None
            }
            
            frame_data[object_id] = properties
            current_frame_objects.append((object_id, properties))
            current_frame_objects.append((object_id, properties))
            
            # Update object track
            if object_id not in self.object_tracks:
                self.object_tracks[object_id] = ObjectTrack(
                    object_id=object_id,
                    label=obj.label,
                    frames_data={}
                )
            
            self.object_tracks[object_id].frames_data[frame_idx] = properties
        
        # Store current frame objects for next frame tracking
        self.previous_objects = current_frame_objects
        
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
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get statistics about object tracking performance."""
        stats = {
            'total_objects': len(self.object_tracks),
            'total_frames': self.num_frames,
            'objects_per_label': {},
            'object_lifespans': {},
            'average_lifespan': 0
        }
        
        total_lifespan = 0
        for obj_id, track in self.object_tracks.items():
            label = track.label
            lifespan = len(track.frames_data)
            
            # Count objects per label
            if label not in stats['objects_per_label']:
                stats['objects_per_label'][label] = 0
            stats['objects_per_label'][label] += 1
            
            # Track lifespans
            stats['object_lifespans'][obj_id] = lifespan
            total_lifespan += lifespan
        
        if len(self.object_tracks) > 0:
            stats['average_lifespan'] = total_lifespan / len(self.object_tracks)
            
        return stats

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
            'num_frames': int(self.num_frames),
            'property_names': self.property_names,
            'object_tracks': {},
            'matrix': {}
        }
        
        # Export object tracks
        for object_id, track in self.object_tracks.items():
            export_data['object_tracks'][object_id] = {
                'object_id': track.object_id,
                'label': track.label,
                'frames_data': self._convert_to_json_serializable(track.frames_data)
            }
        
        # Export matrix (convert frame indices to strings for JSON)
        for frame_idx, frame_data in self.matrix.items():
            export_data['matrix'][str(frame_idx)] = self._convert_to_json_serializable(frame_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported time-series matrix to {output_path}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert NumPy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def export_to_numpy(self, output_path: str):
        """Export the time-series matrix to NumPy format."""
        
        # Filter out non-numeric properties
        numeric_properties = []
        string_properties = []
        
        for prop_name in self.property_names:
            if prop_name in ['label']:  # Known string properties
                string_properties.append(prop_name)
            else:
                numeric_properties.append(prop_name)
        
        # Create dense matrix: [frames, objects, numeric_properties]
        unique_objects = list(self.object_tracks.keys())
        dense_matrix = np.full((self.num_frames, len(unique_objects), len(numeric_properties)), np.nan)
        
        for frame_idx, frame_data in self.matrix.items():
            for obj_idx, object_id in enumerate(unique_objects):
                if object_id in frame_data:
                    for prop_idx, prop_name in enumerate(numeric_properties):
                        if prop_name in frame_data[object_id]:
                            try:
                                value = float(frame_data[object_id][prop_name])
                                dense_matrix[frame_idx, obj_idx, prop_idx] = value
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass
        
        # Collect string properties separately
        string_data = {}
        for prop_name in string_properties:
            string_data[prop_name] = {}
            for frame_idx, frame_data in self.matrix.items():
                string_data[prop_name][str(frame_idx)] = {}
                for object_id in unique_objects:
                    if object_id in frame_data and prop_name in frame_data[object_id]:
                        string_data[prop_name][str(frame_idx)][object_id] = frame_data[object_id][prop_name]
        
        # Save as compressed numpy file
        np.savez_compressed(
            output_path,
            matrix=dense_matrix,
            object_ids=unique_objects,
            numeric_properties=numeric_properties,
            string_properties=string_properties,
            string_data=string_data,
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