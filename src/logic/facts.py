"""
Extract symbolic facts from raw pipeline data using the rule language terms.
"""
import numpy as np
from typing import Dict, List, Any
from src.logic.concepts import *
from src.logic.expressions import *
from src.logic.evaluations import EvaluationContext

class FactsExtractor:
    """Convert raw pipeline data to symbolic facts using language terms"""
    
    def __init__(self):
        self.position_threshold = 30.0  # Distance threshold for "close"
        self.velocity_threshold = 5.0   # Speed threshold for "fast"
        self.change_threshold = 0.1     # Change threshold for "stable"
    
    def analyze_facts_in_results(self, results) -> List[str]:
        """
        Convert all_results to symbolic facts using language terms
        
        Args:
            all_results: Raw pipeline results from video processing
            
        Returns:
            List of symbolic facts expressed in the rule language
        """
        facts = []
        
    
        time_series_matrix = results.get('time_series_matrix')
        causal_edges = results.get('causal_edges', [])
        
        if time_series_matrix:
            # Extract temporal facts
            temporal_facts = self._extract_temporal_facts(time_series_matrix)
            facts.extend(temporal_facts)
            
            # Extract spatial facts  
            spatial_facts = self._extract_spatial_facts(time_series_matrix)
            facts.extend(spatial_facts)
            
            # Extract interaction facts
            interaction_facts = self._extract_interaction_facts(time_series_matrix)
            facts.extend(interaction_facts)
            
            # Extract causal facts
            causal_facts = self._extract_causal_facts(causal_edges, time_series_matrix)
            facts.extend(causal_facts)
        
        return facts
    
    def _extract_temporal_facts(self, time_series_matrix) -> List[str]:
        """Extract facts about temporal behavior using language terms"""
        facts = []
        
        for obj_id, track in time_series_matrix.object_tracks.items():
            frames_data = track.frames_data
            if len(frames_data) < 2:
                continue
                
            # Analyze position changes over time
            positions = []
            velocities = []
            
            for frame_idx in sorted(frames_data.keys()):
                data = frames_data[frame_idx]
                pos = np.array([data['position_x'], data['position_y']])
                vel = np.array([data.get('velocity_x', 0), data.get('velocity_y', 0)])
                positions.append(pos)
                velocities.append(vel)
            
            # Generate facts using language terms
            obj_ref = f"other('{obj_id}')"
            
            # Fact 1: Position evolution pattern
            if self._is_linear_motion(positions):
                facts.append(f"position({obj_ref}).next() ≈ position({obj_ref}).now() + velocity({obj_ref}).now()")
            
            # Fact 2: Velocity stability
            if self._is_stable_velocity(velocities):
                facts.append(f"velocity({obj_ref}).next() ≈ velocity({obj_ref}).now()")
            else:
                facts.append(f"velocity({obj_ref}).next() ≠ velocity({obj_ref}).now()")
            
            # Fact 3: Speed characteristics
            speeds = [np.linalg.norm(vel) for vel in velocities]
            if max(speeds) > self.velocity_threshold:
                facts.append(f"speed({obj_ref}).now() > {self.velocity_threshold}")
            
            # Fact 4: Direction changes
            direction_changes = self._count_direction_changes(velocities)
            if direction_changes > 2:
                facts.append(f"direction_changes({obj_ref}) > 2")
        
        return facts
    
    def _extract_spatial_facts(self, time_series_matrix) -> List[str]:
        """Extract spatial relationship facts"""
        facts = []
        
        # Get all object positions at each frame
        frame_objects = {}
        for obj_id, track in time_series_matrix.object_tracks.items():
            for frame_idx, data in track.frames_data.items():
                if frame_idx not in frame_objects:
                    frame_objects[frame_idx] = {}
                frame_objects[frame_idx][obj_id] = data
        
        # Analyze spatial relationships
        for frame_idx, objects in frame_objects.items():
            obj_list = list(objects.items())
            
            for i, (obj1_id, obj1_data) in enumerate(obj_list):
                for j, (obj2_id, obj2_data) in enumerate(obj_list[i+1:], i+1):
                    
                    # Calculate distance
                    pos1 = np.array([obj1_data['position_x'], obj1_data['position_y']])
                    pos2 = np.array([obj2_data['position_x'], obj2_data['position_y']])
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    # Generate spatial facts
                    obj1_ref = f"other('{obj1_id}')"
                    obj2_ref = f"other('{obj2_id}')"
                    
                    if distance < self.position_threshold:
                        facts.append(f"distance({obj1_ref}.position.now(), {obj2_ref}.position.now()) < {self.position_threshold}")
                        facts.append(f"close_proximity({obj1_ref}, {obj2_ref}) = True")
                    
                    # Relative positioning
                    if abs(pos1[0] - pos2[0]) < 5:  # Vertically aligned
                        facts.append(f"vertically_aligned({obj1_ref}, {obj2_ref}) = True")
                    
                    if abs(pos1[1] - pos2[1]) < 5:  # Horizontally aligned
                        facts.append(f"horizontally_aligned({obj1_ref}, {obj2_ref}) = True")
        
        return facts
    
    def _extract_interaction_facts(self, time_series_matrix) -> List[str]:
        """Extract object interaction facts"""
        facts = []
        
        # Detect following behavior
        for obj1_id, track1 in time_series_matrix.object_tracks.items():
            for obj2_id, track2 in time_series_matrix.object_tracks.items():
                if obj1_id != obj2_id:
                    
                    # Check if obj1 follows obj2's velocity pattern
                    correlation = self._calculate_velocity_correlation(track1, track2)
                    
                    if correlation > 0.8:  # High correlation
                        obj1_ref = f"other('{obj1_id}')"
                        obj2_ref = f"other('{obj2_id}')"
                        facts.append(f"follows({obj1_ref}, {obj2_ref}) = True")
                        facts.append(f"velocity({obj1_ref}).next() ≈ velocity({obj2_ref}).now() * follow_ratio")
                    
                    # Check for avoidance behavior
                    avoidance_detected = self._detect_avoidance_pattern(track1, track2)
                    if avoidance_detected:
                        facts.append(f"avoids({obj1_ref}, {obj2_ref}) = True")
        
        return facts
    
    def _extract_causal_facts(self, causal_edges: List, time_series_matrix) -> List[str]:
        """Extract causal relationship facts using language terms"""
        facts = []
        
        for edge in causal_edges:
            if len(edge) >= 5:
                source, target, lag, edge_type, p_value = edge[:5]
                
                # Convert causal edge to language terms
                source_obj = self._extract_object_from_variable(source)
                target_obj = self._extract_object_from_variable(target)
                source_prop = self._extract_property_from_variable(source)
                target_prop = self._extract_property_from_variable(target)
                
                if source_obj and target_obj and source_prop and target_prop:
                    source_ref = f"other('{source_obj}')" if source_obj != 'self' else "self"
                    target_ref = f"other('{target_obj}')" if target_obj != 'self' else "self"
                    
                    # Generate causal fact
                    causal_fact = f"WHEN {source_prop}({source_ref}).changes() CAUSES {target_prop}({target_ref}).changes() WITH lag={lag} AND strength={1-p_value:.3f}"
                    facts.append(causal_fact)
                    
                    # Generate conditional rule suggestion
                    if p_value < 0.05:  # Statistically significant
                        rule_fact = f"IF {source_prop}({source_ref}).changes() THEN {target_prop}({target_ref}).next() = {target_prop}({target_ref}).now() + influence_from({source_ref})"
                        facts.append(rule_fact)
        
        return facts
    
    def _is_linear_motion(self, positions: List[np.ndarray]) -> bool:
        """Check if motion is approximately linear"""
        if len(positions) < 3:
            return True
            
        # Calculate average direction
        directions = []
        for i in range(1, len(positions)):
            diff = positions[i] - positions[i-1]
            if np.linalg.norm(diff) > 0.1:  # Avoid zero division
                directions.append(diff / np.linalg.norm(diff))
        
        if not directions:
            return True
            
        # Check if directions are consistent
        avg_direction = np.mean(directions, axis=0)
        deviations = [np.linalg.norm(d - avg_direction) for d in directions]
        return np.mean(deviations) < 0.3
    
    def _is_stable_velocity(self, velocities: List[np.ndarray]) -> bool:
        """Check if velocity is approximately stable"""
        if len(velocities) < 2:
            return True
            
        velocity_changes = []
        for i in range(1, len(velocities)):
            change = np.linalg.norm(velocities[i] - velocities[i-1])
            velocity_changes.append(change)
        
        return np.mean(velocity_changes) < self.change_threshold
    
    def _count_direction_changes(self, velocities: List[np.ndarray]) -> int:
        """Count significant direction changes"""
        if len(velocities) < 3:
            return 0
            
        direction_changes = 0
        for i in range(1, len(velocities) - 1):
            prev_vel = velocities[i-1]
            curr_vel = velocities[i]
            next_vel = velocities[i+1]
            
            # Calculate direction change angle
            if np.linalg.norm(prev_vel) > 0.1 and np.linalg.norm(next_vel) > 0.1:
                prev_dir = prev_vel / np.linalg.norm(prev_vel)
                next_dir = next_vel / np.linalg.norm(next_vel)
                
                dot_product = np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                if angle > np.pi / 4:  # More than 45 degrees
                    direction_changes += 1
        
        return direction_changes
    
    def _extract_object_from_variable(self, variable_name: str) -> str:
        """Extract object ID from causal variable name"""
        # Assuming format like "object_0_velocity_x" or "obj_1_position_y"
        parts = variable_name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return None
    
    def _extract_property_from_variable(self, variable_name: str) -> str:
        """Extract property name from causal variable name"""
        if 'position' in variable_name:
            return 'position'
        elif 'velocity' in variable_name:
            return 'velocity'
        elif 'acceleration' in variable_name:
            return 'acceleration'
        return None