"""
Configuration file for CauVid project
Contains dataset paths and other project settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
driving_demo_video_id = "0153f03b-8fbdc1ad"
# driving_demo_video_id = "01118704-e91b1b1c"
DEFAULT_STORAGE_ROOT_CANDIDATES = [
    Path(os.environ["CAUVID_STORAGE_ROOT"]) if os.environ.get("CAUVID_STORAGE_ROOT") else None,
    Path("/storage-02/ml-jsha"),
    Path("/storage-01/ml-jsha/CauVid_Data"),
]
DEFAULT_STORAGE_ROOT = next(
    (path for path in DEFAULT_STORAGE_ROOT_CANDIDATES if path is not None and path.exists()),
    Path("/storage-02/ml-jsha"),
)
DEFAULT_RAW_DRIVING_DATASET = Path(
    os.environ.get(
        "CAUVID_RAW_DRIVING_DATASET",
        DEFAULT_STORAGE_ROOT / "driving-video-with-object-tracking",
    )
)
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("CAUVID_OUTPUT_PATH", PROJECT_ROOT / "output"))
DEFAULT_PIPELINE_OUTPUT_ROOT = Path(
    os.environ.get(
        "CAUVID_PIPELINE_OUTPUT_PATH",
        "/storage-01/ml-jsha/CauVid_output/pipeline_output",
    )
)
DEFAULT_TEMP_ROOT = Path(os.environ.get("CAUVID_TEMP_PATH", PROJECT_ROOT / "temp"))

# Dataset paths
DATASET_PATHS = {
    # Mini driving dataset path
    "driving_mini": Path(os.environ.get("CAUVID_DRIVING_MINI_PATH", PROJECT_ROOT / "dataset" / "driving_mini")),
    # Raw driving dataset path containing .mov videos and mot_labels.csv
    "driving_raw": Path(os.environ.get(
        "CAUVID_RAW_DRIVING_DATASET",
        DEFAULT_RAW_DRIVING_DATASET if DEFAULT_RAW_DRIVING_DATASET.exists() else PROJECT_ROOT / "dataset" / "driving-video-with-object-tracking",
    )),
    # Main dataset directory
    "dataset_root": PROJECT_ROOT / "dataset",
    
    # External dataset directory (for downloaded datasets)
    "external_data": PROJECT_ROOT / "external",
}

# Output paths
OUTPUT_PATHS = {
    # General output directory
    "output": DEFAULT_OUTPUT_ROOT,
    "driving_segmentation_visualization": DEFAULT_OUTPUT_ROOT / "driving_segmentation_visualization",
    "driving_trajectory_visualization": DEFAULT_OUTPUT_ROOT / "driving_trajectory_visualization",
    "driving_trajectory_visualization_smoothed": DEFAULT_OUTPUT_ROOT / "driving_trajectory_visualization_smoothed",
    "driving_trajectory_visualization_with_frames": DEFAULT_OUTPUT_ROOT / "driving_trajectory_visualization_with_frames",
    "driving_primitive_visualization": DEFAULT_OUTPUT_ROOT / "driving_primitive_visualization",
    "driving_ego_primitive_visualization": DEFAULT_OUTPUT_ROOT / "driving_ego_primitive_visualization",
    "driving_seg_feat_vis": DEFAULT_OUTPUT_ROOT / "driving_seg_feat_vis",
    "driving_seg_grps_vis": DEFAULT_OUTPUT_ROOT / "driving_seg_grps_vis",
    # Pipeline output directory
    "pipeline_output": DEFAULT_PIPELINE_OUTPUT_ROOT,
    
    # Temporary output directory
    "temp": DEFAULT_TEMP_ROOT,
}

# Model paths
MODEL_PATHS = {
    # External models directory
    "models": PROJECT_ROOT / "external",
    
    # Depth anything models
    "depth_anything": PROJECT_ROOT / "external" / "Depth-Anything-3",
}

# Config file paths
CONFIG_PATHS = {
    "pattern_mining": PROJECT_ROOT / "configs" / "pattern_mining" / "default.yaml",
    "exp_driving": PROJECT_ROOT / "configs" / "exp_driving" / "default.yaml",
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
    os.makedirs(OUTPUT_PATHS[output_name], exist_ok=True)  # Ensure the output directory exists
    return OUTPUT_PATHS[output_name]

def get_config_path(config_name: str) -> Path:
    """
    Get config file path by name.
    
    Args:
        config_name: Name of the config file entry (e.g., 'pattern_mining')
    
    Returns:
        Path object to the config file
        
    Raises:
        KeyError: If config_name is not found in configuration
    """
    if config_name not in CONFIG_PATHS:
        raise KeyError(f"Config '{config_name}' not found. Available configs: {list(CONFIG_PATHS.keys())}")
    return CONFIG_PATHS[config_name]

def ensure_directories():
    """Create all configured directories if they don't exist"""
    all_paths = {**DATASET_PATHS, **OUTPUT_PATHS, **MODEL_PATHS}
    
    for name, path in all_paths.items():
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)

def get_mini_video_ids():
    folder_path = get_dataset_path('driving_mini') / "videos"
    video_ids = [f.stem for f in folder_path.glob("*.mov")]
    return video_ids

# Ensure directories exist when config is imported
if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    for name, path in DATASET_PATHS.items():
        print(f"  {name}: {path}")
