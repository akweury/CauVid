# CauVid Video Processing Pipeline

This is the first pipeline for the CauVid project that processes video frames from observations and detects objects to create object-centric representations.

## Pipeline Overview

The pipeline consists of the following components:

1. **Frame Loading**: Loads video frames from observation directories
2. **Object Detection**: Uses pretrained models to detect objects in frames
3. **Object-Centric Matrix**: Stores object information frame by frame for temporal analysis

## Features

- ✅ Load video frames from observation directories in `two_parts/observation/`
- ✅ Object detection using YOLOv8 (with fallback to dummy detector)
- ✅ **Specialized circle detection** for synthetic circular objects
- ✅ Object-centric matrix storage for frame-by-frame analysis
- ✅ **Time-series object matrix** for tracking object properties across frames
- ✅ Ground truth integration from metadata files
- ✅ **Precision calculation** with comprehensive metrics (Precision, Recall, F1-Score, Position Error)
- ✅ Export results to JSON format
- ✅ **Numpy array export** for efficient numerical analysis
- ✅ Batch processing support
- ✅ Object tracking across frames

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For GPU acceleration (optional):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Quick Start

```python
from src.video_pipeline import VideoPipeline

# Initialize the pipeline with precision evaluation
pipeline = VideoPipeline("./two_parts", "./output", 
                         detector_type="circle", 
                         position_threshold=20.0)

# Process a single observation with precision calculation
matrix, time_series_matrix = pipeline.process_observation("observation_000000_862365")

# Get results with precision metrics
print(f"Processed {len(matrix.frames_data)} frames")
print(f"Found {len(matrix.object_tracks)} object tracks")
print(f"Overall Precision: {matrix.overall_metrics.precision:.3f}")
print(f"Overall Recall: {matrix.overall_metrics.recall:.3f}")
print(f"Mean Position Error: {matrix.overall_metrics.mean_position_error:.2f} pixels")

# Access time-series matrix
print(f"Time-series dimensions: {time_series_matrix.num_frames} × {len(time_series_matrix.object_tracks)}")
vel_stats = time_series_matrix.get_velocity_statistics()
print(f"Mean speed: {vel_stats['mean_speed']:.2f} px/frame")

# Get object trajectory
first_obj_id = list(time_series_matrix.object_tracks.keys())[0]
trajectory = time_series_matrix.get_object_trajectory(first_obj_id)
print(f"Object {first_obj_id} trajectory: {len(trajectory['frame_indices'])} frames")
```

### Using the Example Script

```bash
python example_usage.py
```

This script provides interactive examples for:
- Single observation processing
- Batch processing multiple observations
- Dependency checking

### Pipeline Components

#### 1. VideoFrameLoader
Handles loading and preprocessing of video frames:

```python
from src.video_pipeline import VideoFrameLoader

loader = VideoFrameLoader("./two_parts")
frames = loader.load_frames("observation_000000_862365")
metadata = loader.load_metadata("observation_000000_862365")
```

#### 2. ObjectDetector
Performs object detection using pretrained models:

```python
from src.video_pipeline import ObjectDetector

detector = ObjectDetector(model_type="yolo")
detected_objects = detector.detect_objects(frame, frame_idx=0)
```

#### 3. ObjectCentricMatrix
Stores and manages object-centric representations:

```python
from src.video_pipeline import ObjectCentricMatrix

matrix = ObjectCentricMatrix()
matrix.add_frame_data(frame_data)
matrix.export_to_json("results.json")
```

## Data Structure

### Input Data Format

The pipeline expects data in the following structure:
```
two_parts/
└── observation/
    ├── observation_000000_862365/
    │   ├── meta.json          # Metadata with ground truth
    │   ├── stats.csv          # Statistics
    │   └── frames/            # Video frames
    │       ├── 000000.png
    │       ├── 000001.png
    │       └── ...
    └── observation_000001_823472/
        └── ...
```

### Output Data Format

The pipeline generates JSON files with object-centric representations:

```json
{
  "frames": {
    "0": {
      "frame_idx": 0,
      "num_objects": 6,
      "objects": [
        {
          "label": "circle_red",
          "confidence": 0.64,
          "center": [75.0, 75.0],
          "bbox": [50, 50, 100, 100],
          "area": 2500.0
        }
      ]
    }
  },
  "precision_metrics": {
    "0": {
      "true_positives": 6,
      "false_positives": 0,
      "false_negatives": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "mean_position_error": 0.90,
      "num_matched_pairs": 6
    }
  },
  "overall_precision": {
    "precision": 1.000,
    "recall": 1.000,
    "f1_score": 1.000,
    "mean_position_error": 0.90
  },
  "tracks": {
    "circle_red_0": [
      {
        "frame_idx": 0,
        "center": [75.0, 75.0],
        "bbox": [50, 50, 100, 100],
        "confidence": 0.64,
        "area": 2500.0
      }
    ]
  },
  "metadata": {
    "num_frames": 60,
    "num_tracks": 12,
    "created_at": "2025-11-10T..."
  }
}
```

## Object Detection Models

### YOLOv8 (Default)
- Fast and accurate object detection
- Supports 80 COCO classes
- GPU accelerated when available

### Custom Models
The pipeline is designed to be extensible. You can add custom object detection models by:

1. Subclassing `ObjectDetector`
2. Implementing the `_load_model()` and `detect_objects()` methods
3. Returning `DetectedObject` instances

```python
class CustomDetector(ObjectDetector):
    def _load_model(self):
        # Load your custom model
        pass
    
    def detect_objects(self, frame, frame_idx):
        # Implement detection logic
        return detected_objects
```

## Configuration

### Pipeline Parameters

- `two_parts_root`: Path to the two_parts directory
- `output_dir`: Directory for saving results
- `detector_type`: Object detection model type ("circle", "yolo", etc.)
- `position_threshold`: Maximum distance in pixels for matching detections to ground truth (default: 20.0)
- `confidence_threshold`: Minimum confidence for object detection (default: 0.5)

### Example Configuration

```python
pipeline = VideoPipeline(
    two_parts_root="./two_parts",
    output_dir="./results",
    detector_type="circle",
    position_threshold=15.0  # Stricter matching criteria
)

# Process with ground truth evaluation
matrix = pipeline.process_observation(
    "observation_000000_862365",
    include_ground_truth=True  # Enable precision calculation
)
```

### Time-Series Object Matrix

The pipeline creates a comprehensive time-series matrix that tracks object properties across all frames:

**Structure**: `[frame_id, object_id, properties]`

**Properties tracked**:
- `position_x`, `position_y`: Object center coordinates
- `velocity_x`, `velocity_y`: Frame-to-frame velocity components  
- `width`, `height`: Bounding box dimensions
- `area`: Object area in pixels
- `confidence`: Detection confidence score
- `color_r`, `color_g`, `color_b`: RGB color values
- `label`: Object classification label

**Access methods**:
```python
# Get trajectory for specific object
trajectory = time_series_matrix.get_object_trajectory("circle_red_000")

# Get all objects in specific frame  
frame_data = time_series_matrix.get_frame_data(frame_idx=10)

# Get property matrix for analysis
position_matrix = time_series_matrix.get_property_matrix('position_x')  # Shape: [frames, objects]

# Export to numpy for numerical analysis
numpy_data = time_series_matrix.export_to_numpy()
```

**Velocity Statistics**:
The matrix automatically calculates velocity statistics:
- Mean velocity components and speed
- Velocity standard deviation  
- Maximum speed across all objects
- Object displacement analysis

### Advanced Feature Extraction

The pipeline extracts sophisticated feature vectors for each object in each frame to enable temporal analysis:

**Feature Vector Components** (`f_t`):
- `mean_velocity_direction`: Object movement direction (radians)
- `mean_speed`: Movement speed (normalized by frame diagonal)
- `contact_pattern`: Proximity score to other objects (0-1)
- `support_pattern`: Stability and position support score (0-1)
- `kinetic_energy`: Normalized kinetic energy based on motion
- `potential_energy`: Normalized potential energy based on height

**Usage**:
```python
# Extract features during processing
results = pipeline.process_observation(
    observation_id,
    extract_features=True,
    analyze_bonds=True
)

# Access feature vectors
object_features = results['object_features']
features_frame_0 = object_features['circle_red_000'][0]  # Features for specific object and frame
feature_vector = features_frame_0.feature_vector  # NumPy array [6,]
```

### Temporal Bond Analysis & Video Segmentation

The pipeline performs sophisticated temporal bond analysis to segment videos based on object behavior consistency:

**Bond Strength Calculation**:
```
bond_strength = cosine_similarity(f_t, f_t+1)

if bond_strength > τ and not key_event_occurs:
    bond_exists = True
else:
    bond_break = True
```

**Key Event Detection**:
Automatically detects significant behavioral changes:
- Direction changes > 60 degrees
- Speed changes > threshold
- Energy changes > threshold

**Video Segmentation**:
- Groups consecutive frames with strong bonds
- Creates segments based on bond breaks
- Analyzes break reasons (threshold, key_event, end_of_video)

**Configuration**:
```python
pipeline = VideoPipeline(
    two_parts_root="./two_parts",
    bond_threshold=0.8,  # Minimum similarity for bonds (τ)
    frame_height=480,    # For feature normalization
    frame_width=640
)
```

**Output Files**:
- `*_features.json`: Feature vectors for all objects and frames
- `*_bonds.json`: Bond analysis with similarity scores and break events
- `*_segments.json`: Video segments with duration and break reasons

### Precision Metrics

The pipeline automatically calculates comprehensive precision metrics when ground truth is available:

- **Precision**: TP / (TP + FP) - Fraction of detections that are correct
- **Recall**: TP / (TP + FN) - Fraction of ground truth objects detected  
- **F1-Score**: Harmonic mean of precision and recall
- **Position Error**: Mean distance between detected and ground truth positions
- **True Positives**: Correctly detected objects
- **False Positives**: Incorrect detections
- **False Negatives**: Missed ground truth objects

## Performance Notes

- **Processing Speed**: ~2-5 FPS depending on hardware and model
- **Memory Usage**: ~100-500MB per observation
- **GPU Acceleration**: Significantly faster with CUDA-enabled PyTorch

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues**
   - Install CUDA-compatible PyTorch version
   - Pipeline will fallback to CPU if GPU unavailable

3. **Memory Issues**
   - Process observations individually instead of batch processing
   - Reduce image resolution if needed

4. **File Not Found Errors**
   - Check that `two_parts` directory structure is correct
   - Verify observation directories contain `frames/` and `meta.json`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

This is the first version of the pipeline. Future enhancements could include:

- [ ] Improved object tracking algorithms
- [ ] Support for additional object detection models
- [ ] Real-time processing capabilities
- [ ] Advanced object relationship analysis
- [ ] Integration with causal inference modules

## Contributing

When extending the pipeline:
1. Follow the existing class structure
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Add unit tests

## License

[Add license information here]