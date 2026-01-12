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
        self.window_size = 5            # Window size for velocity smoothing
    
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
            
            # Print temporal facts immediately after extraction
            if temporal_facts:
                print(f"\n⏰ TEMPORAL FACTS DISCOVERED ({len(temporal_facts)}):")
                for i, fact_data in enumerate(temporal_facts, 1):
                    if isinstance(fact_data, dict):
                        print(f"   {i:2d}. {fact_data['fact']} [confidence: {fact_data['confidence']:.3f}]")
                    else:
                        print(f"   {i:2d}. {fact_data}")
                print()
            
            # Extract spatial facts  
            spatial_facts = self._extract_spatial_facts(time_series_matrix)
            facts.extend(spatial_facts)
            
            # Print spatial facts immediately after extraction
            if spatial_facts:
                print(f"\n📍 SPATIAL FACTS DISCOVERED ({len(spatial_facts)}):")
                for i, fact_data in enumerate(spatial_facts, 1):
                    if isinstance(fact_data, dict):
                        print(f"   {i:2d}. {fact_data['fact']} [confidence: {fact_data['confidence']:.3f}]")
                    else:
                        print(f"   {i:2d}. {fact_data}")
                print()
            
            # Extract interaction facts
            interaction_facts = self._extract_interaction_facts(time_series_matrix)
            facts.extend(interaction_facts)
            
            # Print interaction facts immediately after extraction
            if interaction_facts:
                print(f"\n🤝 INTERACTION FACTS DISCOVERED ({len(interaction_facts)}):")
                for i, fact_data in enumerate(interaction_facts, 1):
                    if isinstance(fact_data, dict):
                        print(f"   {i:2d}. {fact_data['fact']} [confidence: {fact_data['confidence']:.3f}]")
                    else:
                        print(f"   {i:2d}. {fact_data}")
                print()
            
            # Extract causal facts
            causal_facts = self._extract_causal_facts(causal_edges, time_series_matrix)
            facts.extend(causal_facts)
            
            # Print causal facts immediately after extraction
            if causal_facts:
                print(f"\n⚡ CAUSAL FACTS DISCOVERED ({len(causal_facts)}):")
                for i, fact_data in enumerate(causal_facts, 1):
                    if isinstance(fact_data, dict):
                        print(f"   {i:2d}. {fact_data['fact']} [confidence: {fact_data['confidence']:.3f}]")
                    else:
                        print(f"   {i:2d}. {fact_data}")
                print()
        
        return facts
    
    def print_discovered_facts(self, facts: List[str], obs_id: str = None):
        """Print discovered facts in a formatted way.
        
        Args:
            facts: List of symbolic facts to print
            obs_id: Optional observation ID for context
        """
        if obs_id:
            print(f"\n🔍 DISCOVERED FACTS - Observation: {obs_id}")
        else:
            print(f"\n🔍 DISCOVERED FACTS")
        print("=" * 70)
        
        if not facts:
            print("   No facts discovered.")
            return
        
        # Categorize facts for better organization
        temporal_facts = []
        spatial_facts = []
        interaction_facts = []
        causal_facts = []
        other_facts = []
        
        for fact in facts:
            if any(keyword in fact for keyword in ['velocity', 'speed', 'direction_changes', 'position().next()', 'velocity().next()']):
                temporal_facts.append(fact)
            elif any(keyword in fact for keyword in ['distance', 'close_proximity', 'aligned']):
                spatial_facts.append(fact)
            elif any(keyword in fact for keyword in ['follows', 'avoids']):
                interaction_facts.append(fact)
            elif any(keyword in fact for keyword in ['WHEN', 'CAUSES', 'IF', 'THEN']):
                causal_facts.append(fact)
            else:
                other_facts.append(fact)
        
        # Print categorized facts
        if temporal_facts:
            print(f"\n⏰ Temporal Facts ({len(temporal_facts)}):")
            for i, fact in enumerate(temporal_facts, 1):
                print(f"   {i:2d}. {fact}")
        
        if spatial_facts:
            print(f"\n📍 Spatial Facts ({len(spatial_facts)}):")
            for i, fact in enumerate(spatial_facts, 1):
                print(f"   {i:2d}. {fact}")
        
        if interaction_facts:
            print(f"\n🤝 Interaction Facts ({len(interaction_facts)}):")
            for i, fact in enumerate(interaction_facts, 1):
                print(f"   {i:2d}. {fact}")
        
        if causal_facts:
            print(f"\n⚡ Causal Facts ({len(causal_facts)}):")
            for i, fact in enumerate(causal_facts, 1):
                print(f"   {i:2d}. {fact}")
        
        if other_facts:
            print(f"\n📋 Other Facts ({len(other_facts)}):")
            for i, fact in enumerate(other_facts, 1):
                print(f"   {i:2d}. {fact}")
        
        print(f"\n📊 Summary: {len(facts)} total facts discovered")
        print("=" * 70 + "\n")

    def _extract_temporal_facts(self, time_series_matrix) -> List[str]:
        """Extract facts about temporal behavior using language terms"""
        facts = []
        
        for obj_id, track in time_series_matrix.object_tracks.items():
            frames_data = track.frames_data                
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
            obj_ref = f"{obj_id}"
            
            # Fact 1: Position evolution pattern
            position_confidence = self._calculate_linear_motion_confidence(positions)
            facts.append(
                {"fact": f"position({obj_ref}).next() = position({obj_ref}).now() + velocity({obj_ref}).now()",
                 "confidence": position_confidence
                }
            )
            # Fact 2: Velocity stability
            velocity_confidence = self._calculate_stable_velocity_confidence(velocities)
            
            facts.append(
                {"fact": f"velocity({obj_ref}).next() ≈ velocity({obj_ref}).now()",
                 "confidence": velocity_confidence  
                }
            )
            
            
                        
            # Fact 3: Speed characteristics
            speeds = [np.linalg.norm(vel) for vel in velocities]
            speed_confidence = self._calculate_speed_confidence(speeds)
            facts.append(
                {"fact": f"speed({obj_ref}).now() > {self.velocity_threshold}",
                 "confidence": speed_confidence
                }
            )
            
            # Fact 4: Direction changes
            direction_changes, direction_confidence = self._count_direction_changes_with_confidence(velocities)
            facts.append(
                {"fact": f"direction_changes({obj_ref}) > 2",
                 "confidence": direction_confidence
                }
            )
        
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
        
        # Track spatial relationships across frames for confidence calculation
        distance_measurements = {}
        alignment_measurements = {}
        
        # Analyze spatial relationships
        for frame_idx, objects in frame_objects.items():
            obj_list = list(objects.items())
            
            for i, (obj1_id, obj1_data) in enumerate(obj_list):
                for j, (obj2_id, obj2_data) in enumerate(obj_list[i+1:], i+1):
                    
                    # Calculate distance
                    pos1 = np.array([obj1_data['position_x'], obj1_data['position_y']])
                    pos2 = np.array([obj2_data['position_x'], obj2_data['position_y']])
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    # Store measurements for confidence calculation
                    pair_key = tuple(sorted([obj1_id, obj2_id]))
                    if pair_key not in distance_measurements:
                        distance_measurements[pair_key] = []
                        alignment_measurements[pair_key] = {'vertical': [], 'horizontal': []}
                    
                    distance_measurements[pair_key].append(distance)
                    alignment_measurements[pair_key]['vertical'].append(abs(pos1[0] - pos2[0]))
                    alignment_measurements[pair_key]['horizontal'].append(abs(pos1[1] - pos2[1]))
        
        # Generate spatial facts with confidence
        for pair_key, distances in distance_measurements.items():
            obj1_id, obj2_id = pair_key
            obj1_ref = f"{obj1_id}"
            obj2_ref = f"{obj2_id}"
            
            # Distance and proximity facts
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            if min_distance < self.position_threshold:
                # Confidence based on how consistently close they are
                close_frames = sum(1 for d in distances if d < self.position_threshold)
                proximity_confidence = close_frames / len(distances)
                
                facts.append({
                    "fact": f"distance({obj1_ref}.position.now(), {obj2_ref}.position.now()) < {self.position_threshold}",
                    "confidence": proximity_confidence
                })
                facts.append({
                    "fact": f"close_proximity({obj1_ref}, {obj2_ref}) = True", 
                    "confidence": proximity_confidence
                })
            
            # Alignment facts
            vertical_diffs = alignment_measurements[pair_key]['vertical']
            horizontal_diffs = alignment_measurements[pair_key]['horizontal']
            
            # Vertical alignment (x-coordinates similar)
            avg_vertical_diff = np.mean(vertical_diffs)
            if avg_vertical_diff < 5:
                aligned_frames = sum(1 for d in vertical_diffs if d < 5)
                vertical_confidence = aligned_frames / len(vertical_diffs)
                # Boost confidence for smaller average difference
                vertical_confidence = min(1.0, vertical_confidence + (1.0 - avg_vertical_diff / 5) * 0.2)
                
                facts.append({
                    "fact": f"vertically_aligned({obj1_ref}, {obj2_ref}) = True",
                    "confidence": vertical_confidence
                })
            
            # Horizontal alignment (y-coordinates similar)  
            avg_horizontal_diff = np.mean(horizontal_diffs)
            if avg_horizontal_diff < 5:
                aligned_frames = sum(1 for d in horizontal_diffs if d < 5)
                horizontal_confidence = aligned_frames / len(horizontal_diffs)
                # Boost confidence for smaller average difference
                horizontal_confidence = min(1.0, horizontal_confidence + (1.0 - avg_horizontal_diff / 5) * 0.2)
                
                facts.append({
                    "fact": f"horizontally_aligned({obj1_ref}, {obj2_ref}) = True",
                    "confidence": horizontal_confidence
                })
        
        return facts
    
    def _extract_interaction_facts(self, time_series_matrix) -> List[str]:
        """Extract object interaction facts"""
        facts = []
        
        # Detect following behavior
        for obj1_id, track1 in time_series_matrix.object_tracks.items():
            for obj2_id, track2 in time_series_matrix.object_tracks.items():
                if obj1_id != obj2_id:
                    
                    # Check if obj1 follows obj2's velocity pattern
                    correlation, correlation_confidence = self._calculate_velocity_correlation_with_confidence(track1, track2)
                    
                    if correlation > 0.7:  # High correlation threshold
                        obj1_ref = f"{obj1_id}"
                        obj2_ref = f"{obj2_id}"
                        
                        # Following confidence based on correlation strength
                        following_confidence = min(1.0, (correlation - 0.7) / 0.3 * correlation_confidence)
                        
                        facts.append({
                            "fact": f"follows({obj1_ref}, {obj2_ref}) = True",
                            "confidence": following_confidence
                        })
                        facts.append({
                            "fact": f"velocity({obj1_ref}).next() ≈ velocity({obj2_ref}).now() * follow_ratio",
                            "confidence": following_confidence * 0.8  # Slightly lower confidence for derived fact
                        })
                    
                    # # Check for avoidance behavior
                    # avoidance_detected, avoidance_confidence = self._detect_avoidance_pattern_with_confidence(track1, track2)
                    # if avoidance_detected:
                    #     obj1_ref = f"{obj1_id}"
                    #     obj2_ref = f"{obj2_id}"
                    #     facts.append({
                    #         "fact": f"avoids({obj1_ref}, {obj2_ref}) = True",
                    #         "confidence": avoidance_confidence
                    #     })
        
        return facts
    
    def _calculate_velocity_correlation_with_confidence(self, track1, track2) -> tuple:
        """Calculate velocity correlation between two tracks and confidence"""
        # Get synchronized frame data
        common_frames = set(track1.frames_data.keys()) & set(track2.frames_data.keys())
        
        if len(common_frames) < 3:
            return 0.0, 0.0  # Insufficient data
        
        velocities1 = []
        velocities2 = []
        
        for frame_idx in sorted(common_frames):
            data1 = track1.frames_data[frame_idx]
            data2 = track2.frames_data[frame_idx]
            
            vel1 = np.array([data1.get('velocity_x', 0), data1.get('velocity_y', 0)])
            vel2 = np.array([data2.get('velocity_x', 0), data2.get('velocity_y', 0)])
            
            velocities1.append(vel1)
            velocities2.append(vel2)
        
        # Calculate correlation for both x and y components
        vel1_x = [v[0] for v in velocities1]
        vel1_y = [v[1] for v in velocities1]
        vel2_x = [v[0] for v in velocities2]
        vel2_y = [v[1] for v in velocities2]
        
        try:
            # Calculate Pearson correlation
            corr_x = np.corrcoef(vel1_x, vel2_x)[0, 1] if np.std(vel1_x) > 0 and np.std(vel2_x) > 0 else 0
            corr_y = np.corrcoef(vel1_y, vel2_y)[0, 1] if np.std(vel1_y) > 0 and np.std(vel2_y) > 0 else 0
            
            # Handle NaN values
            corr_x = corr_x if not np.isnan(corr_x) else 0
            corr_y = corr_y if not np.isnan(corr_y) else 0
            
            # Average correlation
            avg_correlation = (abs(corr_x) + abs(corr_y)) / 2
            
            # Confidence based on data quantity and correlation consistency
            data_confidence = min(1.0, len(common_frames) / 10)  # More frames = higher confidence
            consistency_confidence = 1.0 - abs(abs(corr_x) - abs(corr_y)) / 2  # Similar x,y correlation = higher confidence
            
            overall_confidence = data_confidence * consistency_confidence
            
            return avg_correlation, overall_confidence
            
        except Exception:
            return 0.0, 0.0
    
    def _detect_avoidance_pattern_with_confidence(self, track1, track2) -> tuple:
        """Detect avoidance behavior between two tracks with confidence"""
        common_frames = set(track1.frames_data.keys()) & set(track2.frames_data.keys())
        
        if len(common_frames) < 5:
            return False, 0.0
        
        avoidance_events = 0
        total_proximity_events = 0
        
        for frame_idx in sorted(common_frames):
            data1 = track1.frames_data[frame_idx]
            data2 = track2.frames_data[frame_idx]
            
            pos1 = np.array([data1['position_x'], data1['position_y']])
            pos2 = np.array([data2['position_x'], data2['position_y']])
            vel1 = np.array([data1.get('velocity_x', 0), data1.get('velocity_y', 0)])
            
            distance = np.linalg.norm(pos1 - pos2)
            
            # Check for close proximity situations
            if distance < self.position_threshold * 1.5:  # Within avoidance range
                total_proximity_events += 1
                
                # Check if velocity points away from other object
                direction_to_other = pos2 - pos1
                if np.linalg.norm(direction_to_other) > 0.1 and np.linalg.norm(vel1) > 0.1:
                    direction_to_other_norm = direction_to_other / np.linalg.norm(direction_to_other)
                    vel1_norm = vel1 / np.linalg.norm(vel1)
                    
                    # If velocity is opposite to direction towards other object, it's avoidance
                    dot_product = np.dot(vel1_norm, direction_to_other_norm)
                    if dot_product < -0.3:  # Moving away
                        avoidance_events += 1
        
        if total_proximity_events == 0:
            return False, 0.0
        
        avoidance_ratio = avoidance_events / total_proximity_events
        
        # Consider it avoidance if ratio is high enough
        if avoidance_ratio > 0.4:
            confidence = min(1.0, avoidance_ratio + (total_proximity_events / 10) * 0.2)
            return True, confidence
        else:
            return False, 0.0
    
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
                    source_ref = f"{source_obj}"
                    target_ref = f"{target_obj}"
                    
                    # Calculate confidence based on statistical significance (p-value)
                    # Lower p-value = higher confidence, with some scaling
                    causal_confidence = max(0.0, min(1.0, 1.0 - p_value))
                    
                    # Boost confidence for very low p-values
                    if p_value < 0.01:
                        causal_confidence = min(1.0, causal_confidence + 0.2)
                    elif p_value < 0.05:
                        causal_confidence = min(1.0, causal_confidence + 0.1)
                    
                    # Generate causal fact
                    causal_fact = f"WHEN {source_prop}({source_ref}).changes() CAUSES {target_prop}({target_ref}).changes() WITH lag={lag} AND strength={1-p_value:.3f}"
                    facts.append({
                        "fact": causal_fact,
                        "confidence": causal_confidence
                    })
                    
                    # Generate conditional rule suggestion
                    if p_value < 0.05:  # Statistically significant
                        rule_fact = f"IF {source_prop}({source_ref}).changes() THEN {target_prop}({target_ref}).next() = {target_prop}({target_ref}).now() + influence_from({source_ref})"
                        # Rule confidence is slightly lower than causal confidence
                        rule_confidence = causal_confidence * 0.9
                        facts.append({
                            "fact": rule_fact,
                            "confidence": rule_confidence
                        })
        
        return facts
    
    def _calculate_linear_motion_confidence(self, positions: List[np.ndarray]) -> float:
        """Calculate confidence score for linear motion (0.0 to 1.0)"""
        if len(positions) < 3:
            return 0.5  # Neutral confidence for insufficient data
            
        # Calculate average direction
        directions = []
        for i in range(1, len(positions)):
            diff = positions[i] - positions[i-1]
            if np.linalg.norm(diff) > 0.1:  # Avoid zero division
                directions.append(diff / np.linalg.norm(diff))
        
        if not directions:
            return 0.5  # Neutral confidence for no movement
            
        # Check consistency of directions
        avg_direction = np.mean(directions, axis=0)
        deviations = [np.linalg.norm(d - avg_direction) for d in directions]
        mean_deviation = np.mean(deviations)
        
        # Convert deviation to confidence (lower deviation = higher confidence)
        # Max deviation is ~2.0, so we normalize and invert
        confidence = max(0.0, min(1.0, 1.0 - (mean_deviation / 2.0)))
        return confidence
    
    def _calculate_stable_velocity_confidence(self, velocities: List[np.ndarray]) -> float:
        """Calculate confidence score for velocity stability using sliding window (0.0 to 1.0)"""
        if len(velocities) < self.window_size:
            return 0.5  # Neutral confidence for insufficient data
            
        # Calculate windowed velocities (average within sliding window)
        windowed_velocities = []
        for i in range(self.window_size - 1, len(velocities)):
            # Get velocities within the window (trailing window)
            window_start = max(0, i - self.window_size + 1)
            window_velocities = velocities[window_start:i + 1]
            # Calculate average velocity for this window
            avg_velocity = np.mean(window_velocities, axis=0)
            windowed_velocities.append(avg_velocity)
        
        if len(windowed_velocities) < 2:
            return 0.5
            
        # Calculate changes between windowed velocities
        velocity_changes = []
        for i in range(1, len(windowed_velocities)):
            change = np.linalg.norm(windowed_velocities[i] - windowed_velocities[i-1])
            velocity_changes.append(change)
        
        mean_change = np.mean(velocity_changes)
        
        # Convert change to confidence (lower change = higher stability confidence)
        # Use change_threshold as reference point, with some scaling factor
        confidence = max(0.0, min(1.0, 1.0 - (mean_change / (self.change_threshold * 10))))
        return confidence
    
    def _calculate_speed_confidence(self, speeds: List[float]) -> float:
        """Calculate confidence that speed exceeds velocity threshold"""
        if not speeds:
            return 0.0
            
        max_speed = max(speeds)
        avg_speed = np.mean(speeds)
        
        # Higher confidence if both max and average speeds exceed threshold
        if max_speed > self.velocity_threshold:
            # Base confidence from max speed ratio
            max_confidence = min(1.0, max_speed / (self.velocity_threshold * 2))
            # Boost confidence if average speed is also high
            avg_boost = min(0.3, avg_speed / self.velocity_threshold * 0.3)
            return min(1.0, max_confidence + avg_boost)
        else:
            # Lower confidence when speeds don't exceed threshold
            return max(0.0, max_speed / self.velocity_threshold * 0.5)
    
    def _count_direction_changes_with_confidence(self, velocities: List[np.ndarray]) -> tuple:
        """Count significant direction changes and return confidence"""
        if len(velocities) < 3:
            return 0, 0.0
            
        direction_changes = 0
        total_checks = 0
        angles = []
        
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
                angles.append(angle)
                total_checks += 1
                
                if angle > np.pi / 4:  # More than 45 degrees
                    direction_changes += 1
        
        # Calculate confidence based on the clarity of direction changes
        if total_checks == 0:
            confidence = 0.0
        elif direction_changes > 2:
            # High confidence if we have many significant changes
            confidence = min(1.0, direction_changes / max(1, total_checks) + 0.3)
        else:
            # Lower confidence for fewer direction changes
            confidence = max(0.0, direction_changes / max(1, total_checks))
            
        return direction_changes, confidence
    
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