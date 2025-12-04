# Video Pipeline Modules

The video processing pipeline has been refactored into clean, modular components:

## Main Pipeline
- **`video_pipeline.py`** - Main orchestration pipeline (clean and readable)

## Core Modules  
- **`data_models.py`** - All data structures and models
- **`frame_loader.py`** - Video frame loading utilities
- **`object_detector.py`** - Object detection implementations
- **`precision_evaluator.py`** - Precision evaluation utilities

## Backup
- **`video_pipeline_backup.py`** - Original monolithic file (2000+ lines)

## Key Benefits
✅ **Clean separation of concerns**  
✅ **Easy to read and maintain**  
✅ **Modular and extensible**  
✅ **Reduced complexity**  
✅ **Better testing capabilities**  

## Usage
The API remains the same - just import from `video_pipeline` as before:

```python
from video_pipeline import VideoPipeline

pipeline = VideoPipeline(
    two_parts_root="../two_parts",
    detector_type="circle"
)

results = pipeline.process_observation("observation_000000_862365")
```

The pipeline automatically imports all necessary components from the modular files.