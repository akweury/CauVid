"""
Configuration file for CauVid project
Contains dataset paths and other project settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Dataset paths
DATASET_PATHS = {
    # Mini driving dataset path
    "driving_mini": PROJECT_ROOT / "dataset" / "driving_mini",

    # Main dataset directory
    "dataset_root": PROJECT_ROOT / "dataset",
    
    # External dataset directory (for downloaded datasets)
    "external_data": PROJECT_ROOT / "external",
}

# Output paths
OUTPUT_PATHS = {
    # General output directory
    "output": PROJECT_ROOT / "output",
    
    # Pipeline output directory
    "pipeline_output": PROJECT_ROOT / "pipeline_output",
    
    # Temporary output directory
    "temp": PROJECT_ROOT / "temp",
}

# Model paths
MODEL_PATHS = {
    # External models directory
    "models": PROJECT_ROOT / "external",
    
    # Depth anything models
    "depth_anything": PROJECT_ROOT / "external" / "Depth-Anything-3",
}

def get_dataset_path(dataset_name: str) -> Path:
    """
    Get dataset path by name
    
    Args:
        dataset_name: Name of the dataset (e.g., 'driving_mini', 'raw_video_frames')
    
    Returns:
        Path object to the dataset
        
    Raises:
        KeyError: If dataset_name is not found in configuration
    """
    if dataset_name not in DATASET_PATHS:
        raise KeyError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_PATHS.keys())}")
    
    return DATASET_PATHS[dataset_name]

def get_output_path(output_name: str) -> Path:
    """
    Get output path by name
    
    Args:
        output_name: Name of the output directory (e.g., 'output', 'pipeline_output')
    
    Returns:
        Path object to the output directory
        
    Raises:
        KeyError: If output_name is not found in configuration
    """
    if output_name not in OUTPUT_PATHS:
        raise KeyError(f"Output path '{output_name}' not found. Available outputs: {list(OUTPUT_PATHS.keys())}")
    
    return OUTPUT_PATHS[output_name]

def ensure_directories():
    """Create all configured directories if they don't exist"""
    all_paths = {**DATASET_PATHS, **OUTPUT_PATHS, **MODEL_PATHS}
    
    for name, path in all_paths.items():
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)

# Ensure directories exist when config is imported
if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    for name, path in DATASET_PATHS.items():
        print(f"  {name}: {path}")





