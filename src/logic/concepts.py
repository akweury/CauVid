
"""
Core concepts for the object motion rule discovery language.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ObjectState:
    """Represents the state of an object at a specific time"""
    object_id: str
    frame_idx: int
    position_x: float
    position_y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0

class TemporalReference:
    """Base class for temporal references in the language"""
    pass

class Now(TemporalReference):
    """Current time step"""
    def __str__(self):
        return "now()"

class Next(TemporalReference):
    """Next time step"""
    def __str__(self):
        return "next()"

class Prev(TemporalReference):
    """Previous time step"""
    def __str__(self):
        return "prev()"

class At(TemporalReference):
    """Specific time step"""
    def __init__(self, time_step: int):
        self.time_step = time_step
    
    def __str__(self):
        return f"at({self.time_step})"

class Property(ABC):
    """Base class for object properties"""
    
    def __init__(self, object_ref: 'ObjectReference', temporal_ref: TemporalReference):
        self.object_ref = object_ref
        self.temporal_ref = temporal_ref
    
    @abstractmethod
    def evaluate(self, context: 'EvaluationContext') -> Union[float, np.ndarray]:
        """Evaluate this property in the given context"""
        pass
    
    def __str__(self):
        return f"{self.object_ref}.{self.__class__.__name__.lower()}.{self.temporal_ref}"

class Position(Property):
    """Object position property"""
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        """Returns [x, y] position"""
        state = context.get_object_state(self.object_ref, self.temporal_ref)
        return np.array([state.position_x, state.position_y])

class Velocity(Property):
    """Object velocity property"""
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        """Returns [vx, vy] velocity"""
        state = context.get_object_state(self.object_ref, self.temporal_ref)
        return np.array([state.velocity_x, state.velocity_y])

class Acceleration(Property):
    """Object acceleration property"""
    
    def evaluate(self, context: 'EvaluationContext') -> np.ndarray:
        """Returns [ax, ay] acceleration"""
        state = context.get_object_state(self.object_ref, self.temporal_ref)
        return np.array([state.acceleration_x, state.acceleration_y])

class Speed(Property):
    """Object speed (velocity magnitude)"""
    
    def evaluate(self, context: 'EvaluationContext') -> float:
        """Returns speed scalar"""
        state = context.get_object_state(self.object_ref, self.temporal_ref)
        return np.sqrt(state.velocity_x**2 + state.velocity_y**2)

class ObjectReference:
    """Reference to an object in the scene"""
    pass

class Self(ObjectReference):
    """Reference to current object"""
    def __str__(self):
        return "self"

class Other(ObjectReference):
    """Reference to specific other object"""
    def __init__(self, object_id: str):
        self.object_id = object_id
    
    def __str__(self):
        return f"other({self.object_id})"

class Nearest(ObjectReference):
    """Reference to nearest object"""
    def __str__(self):
        return "nearest()"

# Convenience functions for creating properties
def position_now(obj_ref: ObjectReference = None) -> Position:
    """Current position of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Position(obj_ref, Now())

def position_next(obj_ref: ObjectReference = None) -> Position:
    """Next position of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Position(obj_ref, Next())

def position_prev(obj_ref: ObjectReference = None) -> Position:
    """Previous position of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Position(obj_ref, Prev())

def velo_now(obj_ref: ObjectReference = None) -> Velocity:
    """Current velocity of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Velocity(obj_ref, Now())

def velo_next(obj_ref: ObjectReference = None) -> Velocity:
    """Next velocity of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Velocity(obj_ref, Next())

def velo_prev(obj_ref: ObjectReference = None) -> Velocity:
    """Previous velocity of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Velocity(obj_ref, Prev())

def speed_now(obj_ref: ObjectReference = None) -> Speed:
    """Current speed of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Speed(obj_ref, Now())

def acceleration_now(obj_ref: ObjectReference = None) -> Acceleration:
    """Current acceleration of object"""
    if obj_ref is None:
        obj_ref = Self()
    return Acceleration(obj_ref, Now())



