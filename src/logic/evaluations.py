"""
Evaluation context for rule execution.
"""
from typing import Dict, List, Optional
import numpy as np
from src.logic.concepts import *

class EvaluationContext:
    """Context for evaluating rules and expressions"""
    
    def __init__(self, current_frame: int, current_object: str):
        self.current_frame = current_frame
        self.current_object = current_object
        self.object_states: Dict[int, Dict[str, ObjectState]] = {}  # frame -> object_id -> state
        self.predicted_states: Dict[int, Dict[str, ObjectState]] = {}  # for future states
    
    def add_object_state(self, frame: int, object_id: str, state: ObjectState):
        """Add object state for a specific frame"""
        if frame not in self.object_states:
            self.object_states[frame] = {}
        self.object_states[frame][object_id] = state
    
    def get_object_state(self, obj_ref: ObjectReference, temp_ref: TemporalReference) -> ObjectState:
        """Get object state based on reference and time"""
        
        # Resolve object reference
        if isinstance(obj_ref, Self):
            object_id = self.current_object
        elif isinstance(obj_ref, Other):
            object_id = obj_ref.object_id
        elif isinstance(obj_ref, Nearest):
            object_id = self._find_nearest_object()
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
        
        # Resolve temporal reference
        if isinstance(temp_ref, Now):
            frame = self.current_frame
        elif isinstance(temp_ref, Next):
            frame = self.current_frame + 1
        elif isinstance(temp_ref, Prev):
            frame = self.current_frame - 1
        else:
            raise ValueError(f"Unknown temporal reference: {temp_ref}")
        
        # Get state from appropriate storage
        if frame <= self.current_frame:
            # Historical/current state
            if frame in self.object_states and object_id in self.object_states[frame]:
                return self.object_states[frame][object_id]
        else:
            # Future state (predicted)
            if frame in self.predicted_states and object_id in self.predicted_states[frame]:
                return self.predicted_states[frame][object_id]
        
        raise ValueError(f"No state available for object {object_id} at frame {frame}")
    
    def set_property_value(self, property: Property, value):
        """Set property value (used for assignments)"""
        obj_ref = property.object_ref
        temp_ref = property.temporal_ref
        
        # Resolve references
        if isinstance(obj_ref, Self):
            object_id = self.current_object
        elif isinstance(obj_ref, Other):
            object_id = obj_ref.object_id
        else:
            raise ValueError("Can only assign to self or specific other objects")
        
        if isinstance(temp_ref, Next):
            frame = self.current_frame + 1
        else:
            raise ValueError("Can only assign to next time step")
        
        # Ensure predicted state exists
        if frame not in self.predicted_states:
            self.predicted_states[frame] = {}
        if object_id not in self.predicted_states[frame]:
            # Copy current state as base for prediction
            current_state = self.get_object_state(Self() if object_id == self.current_object else Other(object_id), Now())
            self.predicted_states[frame][object_id] = ObjectState(
                object_id=object_id,
                frame_idx=frame,
                position_x=current_state.position_x,
                position_y=current_state.position_y,
                velocity_x=current_state.velocity_x,
                velocity_y=current_state.velocity_y,
                acceleration_x=current_state.acceleration_x,
                acceleration_y=current_state.acceleration_y
            )
        
        # Set the property value
        state = self.predicted_states[frame][object_id]
        if isinstance(property, Position):
            if isinstance(value, np.ndarray) and len(value) == 2:
                state.position_x, state.position_y = value
            else:
                raise ValueError("Position must be 2D array")
        elif isinstance(property, Velocity):
            if isinstance(value, np.ndarray) and len(value) == 2:
                state.velocity_x, state.velocity_y = value
            else:
                raise ValueError("Velocity must be 2D array")
        # Add more property types as needed
    
    def _find_nearest_object(self) -> str:
        """Find nearest object to current object"""
        if self.current_frame not in self.object_states:
            raise ValueError("No objects in current frame")
        
        current_pos = self.get_object_state(Self(), Now())
        current_position = np.array([current_pos.position_x, current_pos.position_y])
        
        min_distance = float('inf')
        nearest_id = None
        
        for obj_id, state in self.object_states[self.current_frame].items():
            if obj_id != self.current_object:
                obj_position = np.array([state.position_x, state.position_y])
                distance = np.linalg.norm(current_position - obj_position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = obj_id
        
        if nearest_id is None:
            raise ValueError("No other objects found")
        
        return nearest_id