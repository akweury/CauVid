"""
Simple interface for getting time_series_matrix from video processing pipeline.
"""

import sys
from pathlib import Path
import numpy as np

import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite import plotting as tp
from tigramite.jpcmciplus import JPCMCIplus

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
        two_parts_root=two_parts_root, output_dir="./output", detector_type="circle"
    )

    # Process observation
    results = pipeline.process_observation(
        observation_id=observation_id,
        include_ground_truth=True,
        extract_features=False,
        analyze_bonds=False,
    )

    return results["time_series_matrix"]


def process_and_get_matrix(observation_id: str):
    """
    Call the previous function and get the time_series_matrix.
    This function concatenates all object features and runs PCMCI on the combined dataset.

    Args:
        observation_id: ID of the observation to process

    Returns:
        TimeSeriesObjectMatrix object
    """
    time_series_matrix = get_time_series_matrix(observation_id)
    len_time_series = time_series_matrix.num_frames
    object_labels = list(time_series_matrix.matrix[0].keys())
    feature_names = time_series_matrix.property_names
    exclude_features = ["label"]

    dataset = []
    for i in range(time_series_matrix.num_frames):
        current_matrix = time_series_matrix.matrix[i]
        datapoint = []
        for object_i in object_labels:
            for feature_i in feature_names:
                if feature_i in exclude_features:
                    continue
                datapoint.append(current_matrix[object_i][feature_i])
        dataset.append(datapoint)

    var_names = [
        f"{obj}_{feat}"
        for obj in object_labels
        for feat in feature_names
        if feat not in exclude_features
    ]
    var_name_tuples = [
        (obj, feat)
        for obj in object_labels
        for feat in feature_names
        if feat not in exclude_features
    ]

    dataframe = pp.DataFrame(
        data=np.array(dataset),
        var_names=var_names,
    )

    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    results = pcmci.run_pcmci(tau_max=1, pc_alpha=0.05)

    graph = results["graph"]
    p_matrix = results["p_matrix"]

    edges = []
    for i, j, tau in zip(*np.where(graph != "")):
        if tau > 0:
            edges.append(
                (
                    str(var_name_tuples[i]),
                    str(var_name_tuples[j]),
                    f"{tau}",
                    graph[i, j, tau],
                    p_matrix[i, j, tau],
                )
            )

    # edges: list of (source_variable, target_variable, time_lag, edge_type, p_value)
    # where source_variable and target_variable are in the format (object_label, feature_name)
    # time_lag: time between source and target (currently removing instantaneous edges with lag 0, can be changed by not checking tau > 0)
    # edge_type: type of causal relationship
    # p_value: significance of the edge (lower is more significant)
    return edges


def process_and_get_matrix_v2(observation_id: str):
    """
    Call the previous function and get the time_series_matrix.
    This function processes each object individually and runs PCMCI on each object's features separately.
    I.e., there is a time series per object, and the features of that object are the variables.
    In the current form, this does not capture inter-object causal relationships.

    Args:
        observation_id: ID of the observation to process

    Returns:
        TimeSeriesObjectMatrix object
    """
    time_series_matrix = get_time_series_matrix(observation_id)
    len_time_series = time_series_matrix.num_frames
    object_labels = list(time_series_matrix.matrix[0].keys())
    feature_names = time_series_matrix.property_names
    exclude_features = ["label"]

    dataset = []
    for object_i in object_labels:
        object_dataset = []
        for i in range(time_series_matrix.num_frames):
            current_matrix = time_series_matrix.matrix[i]
            datapoint = []
            for feature_i in feature_names:
                if feature_i in exclude_features:
                    continue
                datapoint.append(current_matrix[object_i][feature_i])
            object_dataset.append(datapoint)
        dataset.append(object_dataset)

    var_names = [feat for feat in feature_names if feat not in exclude_features]

    # datatset: datasets / timesteps / variables
    dataframe = pp.DataFrame(
        data=np.array(dataset),
        var_names=var_names,
        analysis_mode="multiple",
    )

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=CMIsymb(), verbosity=0)

    results = pcmci.run_pcmci(tau_max=1, pc_alpha=0.05)

    graph = results["graph"]
    p_matrix = results["p_matrix"]

    edges = []
    for i, j, tau in zip(*np.where(graph != "")):
        if tau > 0:
            edges.append(
                (
                    str(var_names[i]),
                    str(var_names[j]),
                    f"{tau}",
                    graph[i, j, tau],
                    p_matrix[i, j, tau],
                )
            )

    # edges: list of (source_variable, target_variable, time_lag, edge_type, p_value)
    # where source_variable and target_variable are in the format (object_label, feature_name)
    # time_lag: time between source and target (currently removing instantaneous edges with lag 0, can be changed by not checking tau > 0)
    # edge_type: type of causal relationship
    # p_value: significance of the edge (lower is more significant)
    return edges


if __name__ == "__main__":
    # Example usage
    obs_id = "observation_000000_862365"
    matrix = process_and_get_matrix_v2(obs_id)
    print(f"Processed time series matrix for observation {obs_id}")
