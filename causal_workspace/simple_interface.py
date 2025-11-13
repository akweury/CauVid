"""
Simple interface for getting time_series_matrix from video processing pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path for pipeline imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from video_pipeline import VideoPipeline


def get_time_series_matrix(observation_id: str, two_parts_root: str = "../two_parts"):
    """
    Get the time_series_matrix for a given observation.
    
    Args:
        observation_id: ID of the observation to process
        two_parts_root: Path to the two_parts data directory
        
    Returns:
        TimeSeriesObjectMatrix object containing the processed time series data
    """
    # Fix the path to be relative to the causal_workspace directory
    if two_parts_root == "../two_parts":
        two_parts_root = str(Path(__file__).parent.parent / "two_parts")
    # Initialize pipeline
    pipeline = VideoPipeline(
        two_parts_root=two_parts_root,
        output_dir="./output",
        detector_type="circle"
    )
    
    # Process observation
    results = pipeline.process_observation(
        observation_id=observation_id,
        include_ground_truth=True,
        extract_features=False,
        analyze_bonds=False
    )
    
    return results['time_series_matrix']


def process_and_get_matrix(observation_id: str):
    """
    Call the previous function and get the time_series_matrix.
    
    Args:
        observation_id: ID of the observation to process
        
    Returns:
        TimeSeriesObjectMatrix object
    """
    time_series_matrix = get_time_series_matrix(observation_id)
    
    # TODO: Add your causal analysis code here
    
    return time_series_matrix


if __name__ == "__main__":
    # Example usage
    obs_id = "observation_000000_862365"
    matrix = process_and_get_matrix(obs_id)
    print(f"Processed time series matrix for observation {obs_id}")
    