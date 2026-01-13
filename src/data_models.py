"""
Data structures and models for the video processing pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]  # x, y center coordinates
    area: float
    frame_idx: int
    rgb_color: Optional[Tuple[int, int, int]] = None  # Most common RGB color (0-255 range)


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
class FrameFeatureVector:
    """Represents extracted features for an object in a specific frame."""
    object_id: str
    frame_idx: int
    
    # Basic features
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    
    # Motion features
    mean_velocity_direction: float  # Direction of average velocity
    mean_speed: float              # Average speed magnitude
    
    # Interaction features
    contact_pattern: float         # Pattern of contact with other objects
    support_pattern: float         # Pattern of support relationships
    
    # Energy features
    kinetic_energy: float          # Kinetic energy based on motion
    potential_energy: float        # Potential energy based on position


@dataclass
class BondStrength:
    """Represents the strength of a bond between objects across frames."""
    object_id_a: str
    object_id_b: str
    frame_t: int
    frame_t1: int
    bond_strength: float  # Cosine similarity between feature vectors
    key_event_detected: bool = False


@dataclass
class BondType:
    """Represents a classified type of bond between objects."""
    bond_type_id: str
    description: str
    pattern_signature: Dict[str, float]  # Feature pattern that defines this bond type
    example_bonds: List[BondStrength]
    frequency: int  # How often this bond type occurs


@dataclass
class ClassifiedBond:
    """A bond that has been classified into a specific type."""
    bond: BondStrength
    bond_type: BondType
    classification_confidence: float


@dataclass
class VideoSegment:
    """Represents a segment of the video based on bond analysis."""
    segment_id: str
    start_frame: int
    end_frame: int
    dominant_bond_type: Optional[BondType]
    bond_break_points: List[int]  # Frames where significant bond breaks occur
    segment_summary: str


@dataclass
class PrecisionMetrics:
    """Precision evaluation metrics for object detection."""
    precision: float
    recall: float
    f1_score: float
    mean_position_error: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class FrameData:
    """Container for all data related to a single frame."""
    frame_idx: int
    detected_objects: List[DetectedObject]
    ground_truth_objects: Optional[List[DetectedObject]] = None
    precision_metrics: Optional[PrecisionMetrics] = None
    frame_size: Optional[Tuple[int, int]] = None  # width, height