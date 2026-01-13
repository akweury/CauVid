"""
CauVid Video Processing Pipeline

Main entry point for the video processing pipeline.
Orchestrates causal analysis and safety pattern detection for traffic scenarios.
"""

import logging
from typing import Dict, List, Any

# Import from modular components
from time_series_matrix import TimeSeriesObjectMatrix
from video_pipeline_core import VideoPipeline

from causal_workspace.simple_interface import process_and_get_matrix_v2
import data_process 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_object_behaviors(time_series_matrix: TimeSeriesObjectMatrix) -> Dict[str, Any]:
    """
    Analyze and symbolically describe the temporal behaviors of all objects.
    
    This function converts raw trajectory data into symbolic behavior descriptions
    that can be used for safety pattern analysis and causal interpretation.
    
    Args:
        time_series_matrix: Time-series object matrix with object trajectories
        
    Returns:
        Dictionary containing symbolic behavior descriptions for each object
        
    Example output structure:
    {
        'object_0_001': {
            'movement_type': 'linear_motion',
            'speed_profile': 'constant_speed',
            'direction_changes': 2,
            'acceleration_events': ['deceleration_at_frame_15'],
            'spatial_bounds': {'min_x': 100, 'max_x': 200, 'min_y': 50, 'max_y': 150},
            'interaction_events': ['close_approach_with_object_1_at_frame_20'],
            'behavior_phases': [
                {'phase': 'approach', 'frames': [0, 15], 'description': 'moving_right_constant_speed'},
                {'phase': 'decelerate', 'frames': [15, 25], 'description': 'slowing_down_near_object_1'},
                {'phase': 'avoid', 'frames': [25, 40], 'description': 'changing_direction_upward'}
            ]
        },
        'object_1_002': {
            'movement_type': 'curved_motion',
            'speed_profile': 'accelerating',
            'direction_changes': 1,
            'acceleration_events': ['acceleration_at_frame_10'],
            'spatial_bounds': {'min_x': 150, 'max_x': 300, 'min_y': 100, 'max_y': 200},
            'interaction_events': ['collision_risk_with_object_0_at_frame_20'],
            'behavior_phases': [
                {'phase': 'entry', 'frames': [0, 10], 'description': 'entering_scene_from_left'},
                {'phase': 'accelerate', 'frames': [10, 30], 'description': 'accelerating_toward_center'},
                {'phase': 'maintain', 'frames': [30, 50], 'description': 'maintaining_trajectory'}
            ]
        },
        'summary': {
            'total_objects': 2,
            'total_interactions': 1,
            'dominant_behavior': 'avoidance_scenario',
            'scenario_type': 'multi_object_interaction'
        }
    }
    """
    raise NotImplementedError("Object behavior analysis not yet implemented")

def analyze_safety_patterns(causal_edges: List, time_series_matrix: TimeSeriesObjectMatrix) -> Dict[str, Any]:
    """
    Analyze safety patterns from causal relationships and time-series data.
    
    Args:
        causal_edges: List of discovered causal relationships
        time_series_matrix: Time-series object matrix
        
    Returns:
        Dictionary containing identified safety patterns
        
        
    # Example output structure:
{
    'high_speed_approach': {
        'frequency': 3,
        'risk_score': 0.8,
        'examples': [...]
    },
    'collision_risk': {
        'frequency': 2, 
        'risk_score': 0.9,
        'examples': [...]
    },
    'lane_change_cascade': {
        'frequency': 1,
        'risk_score': 0.6,
        'examples': [...]
    }
}
    """
    raise NotImplementedError("Safety pattern analysis not yet implemented")


def build_prediction_models(all_results: Dict[str, Any], all_causal_graphs: Dict[str, List]) -> Dict[str, Any]:
    """
    Build predictive models using causal relationships across all observations.
    
    Args:
        all_results: Dictionary of results from all processed observations
        all_causal_graphs: Dictionary of causal graphs for all observations
        
    Returns:
        Dictionary containing prediction model results and accuracies
    """
    raise NotImplementedError("Prediction model building not yet implemented")


def validate_pipeline_results(all_results: Dict[str, Any], all_causal_graphs: Dict[str, List]) -> Dict[str, Any]:
    """
    Validate pipeline results for consistency and accuracy.
    
    Args:
        all_results: Dictionary of results from all processed observations
        all_causal_graphs: Dictionary of causal graphs for all observations
        
    Returns:
        Dictionary containing validation metrics and results
    """
    raise NotImplementedError("Pipeline validation not yet implemented")


def initialize_pipeline():
    """Initialize the video processing pipeline.
    
    Returns:
        tuple: (pipeline, observation_ids)
    """
    logger.info("=== CauVid Complete Pipeline ===")
    
    # Initialize pipeline
    pipeline = VideoPipeline(
        two_parts_root="./two_parts",
        output_dir="./pipeline_output",
        detector_type="circle",  # or "yolo"
        position_threshold=20.0
    )
    
    # Get available observations
    observation_ids = pipeline.frame_loader.get_observation_ids()
    
    if not observation_ids:
        logger.warning("No observations found in the data directory")
        return None, []
    
    logger.info(f"Found {len(observation_ids)} observations to process")
    return pipeline, observation_ids


def display_dataset_overview(observation_ids: List[str]):
    """Display comprehensive dataset overview.
    
    Args:
        observation_ids: List of observation IDs to process
    """
    print("\n" + "="*100)
    print("🎬 CAUVID VIDEO PROCESSING PIPELINE - DATASET OVERVIEW")
    print("="*100)
    
    print(f"📁 Dataset Information:")
    print(f"   • Total Observations: {len(observation_ids)}")
    print(f"   • Data Source: ./two_parts")
    print(f"   • Output Directory: ./pipeline_output")
    print(f"   • Detection Method: circle detector")
    print(f"   • Position Threshold: 20.0 pixels")
    
    print(f"\n🎯 Observation IDs to Process:")
    for i, obs_id in enumerate(observation_ids, 1):
        status_icon = "🎦"
        print(f"   {i:2d}. {status_icon} {obs_id}")
    
    print(f"\n🔧 Pipeline Configuration:")
    print(f"   • Ground Truth: ✓ Enabled")
    print(f"   • Feature Extraction: ❌ Disabled")
    print(f"   • Bond Analysis: ❌ Disabled")
    print(f"   • Causal Analysis: ✓ Enabled")
    print(f"   • Symbolic Facts: ✓ Enabled")
    print(f"   • Reasoning Rules: ✓ Enabled")
    
    print(f"\n🚀 Processing Pipeline Steps:")
    print(f"   1. 📹 Video frame loading and object detection")
    print(f"   2. 📊 Trajectory segmentation and bond analysis")
    print(f"   3. 🧠 Symbolic fact extraction")
    print(f"   4. 📋 Reasoning rule generation")  
    print(f"   5. 🔗 Causal relationship discovery")
    print(f"   6. 📈 Cross-observation analysis")
    print(f"   7. 🎯 Safety pattern identification")
    
    print("="*100)
    print("🏁 Starting processing pipeline...\n")


def log_obs_summary(obs_id: str, processed_observation: Dict[str, Any]):
    """Log a detailed summary of processing results with data types and shapes.
    
    Args:
        obs_id: Observation ID
        processed_observation: Processing results dictionary
    """
    print(f"\n🔍 RESULTS SUMMARY - {obs_id}")
    print("-" * 60)
    
    if not isinstance(processed_observation, dict):
        print(f"   Type: {type(processed_observation).__name__}, Value: {processed_observation}")
        return
    
    for key, value in processed_observation.items():
        value_type = type(value).__name__
        
        # Special handling for time_series_matrix
        if key == 'time_series_matrix' and hasattr(value, '__dict__'):
            print(f"   {key:20s}: {value_type:15s} | TimeSeriesMatrix object")
            # Show internal attributes of time_series_matrix
            for attr_name in dir(value):
                if not attr_name.startswith('_') and not callable(getattr(value, attr_name)):
                    attr_value = getattr(value, attr_name, None)
                    if attr_value is not None:
                        attr_type = type(attr_value).__name__
                        if hasattr(attr_value, 'shape'):
                            attr_info = f"shape={attr_value.shape}, dtype={getattr(attr_value, 'dtype', 'unknown')}"
                        elif isinstance(attr_value, (list, dict)):
                            attr_info = f"len={len(attr_value)}"
                        else:
                            attr_info = f"value={str(attr_value)[:30]}{'...' if len(str(attr_value)) > 30 else ''}"
                        print(f"     └─ {attr_name:16s}: {attr_type:13s} | {attr_info}")
            continue
        
        # Get detailed information based on data type
        if hasattr(value, 'shape'):  # numpy arrays, tensors
            info = f"shape={value.shape}, dtype={getattr(value, 'dtype', 'unknown')}"
        elif isinstance(value, (list, tuple)):
            if len(value) > 0 and hasattr(value[0], 'shape'):
                info = f"len={len(value)}, first_item_shape={value[0].shape}"
            else:
                info = f"len={len(value)}"
        elif isinstance(value, dict):
            info = f"keys={len(value.keys())}, keys={list(value.keys())[:3]}{'...' if len(value.keys()) > 3 else ''}"
        elif isinstance(value, str):
            info = f"len={len(value)}"
        elif hasattr(value, '__len__'):
            info = f"len={len(value)}"
        else:
            info = f"value={str(value)[:50]}{'...' if len(str(value)) > 50 else ''}"
        
        print(f"   {key:20s}: {value_type:15s} | {info}")
    
    print("-" * 60)


def process_single_observation(pipeline: VideoPipeline, obs_id: str) -> Dict[str, Any]:
    """Process a single observation through the pipeline.
    
    Args:
        pipeline: Initialized VideoPipeline instance
        obs_id: Observation ID to process
        
    Returns:
        Dictionary containing processing results
    """
    logger.info(f"Processing observation: {obs_id}")
    
    # Process the observation
    processed_observation = pipeline.process_observation(
        observation_id=obs_id,
        include_ground_truth=True,
        extract_features=False,
        analyze_bonds=False
    )
    
    # Log results summary
    log_obs_summary(obs_id, processed_observation)
    
    # Extract symbolic facts
    from logic.facts import FactsExtractor
    facts_extractor = FactsExtractor()
    symbolic_facts = facts_extractor.analyze_facts_in_results(processed_observation)
    # Store local variables for logging
    processed_observation["_symbolic_facts"] = symbolic_facts
    
    return processed_observation

def process_single_observation_logic(pipeline: VideoPipeline, obs_id: str) -> Dict[str, Any]:
    """Process a single observation through the pipeline.
    
    Args:
        pipeline: Initialized VideoPipeline instance
        obs_id: Observation ID to process
        
    Returns:
        Dictionary containing processing results
    """
    logger.info(f"Processing observation: {obs_id}")
    
    
    
    # Process the observation
    obs_raw = pipeline.process_observation(
        observation_id=obs_id,
        include_ground_truth=True,
        extract_features=False,
        analyze_bonds=False
    )
    # Log results summary
    log_observation_summary(obs_id, obs_raw)
    
    from logic.facts import FactsExtractor
    facts_extractor = FactsExtractor()
    kinematics_facts = facts_extractor.analyze_facts_in_results(obs_raw)
    
        
    # smooth the observation data
    ts_matrix = list(obs_raw["time_series_matrix"].matrix.values())
    processed_ts_matrix = data_process.window_smooth(ts_matrix, window_size=3)

    # Extract symbolic facts
    from logic.clauses import learn_rules
    symbolic_facts = learn_rules(processed_ts_matrix)    
    
    facts = {
        "kinematics_facts": kinematics_facts,
        "symbolic_facts": symbolic_facts
    }
    
    return facts  


def log_observation_summary(obs_id: str, processed_observation: Dict[str, Any]):
    """Log detailed processing summary for current observation.
    
    Args:
        obs_id: Current observation ID
        processed_observation: Processing results for this observation
        observation_ids: List of all observation IDs for progress calculation
    """
    print("\n" + "="*80)
    print(f"📊 OBSERVATION {obs_id} - PROCESSING COMPLETE")
    print("="*80)
    
    # Dataset overview for current observation
    time_series_matrix = processed_observation.get("time_series_matrix")
    total_frames = "N/A"
    detected_objects = 0
    
    # Extract information from time_series_matrix if available
    if time_series_matrix and hasattr(time_series_matrix, '__dict__'):
        if hasattr(time_series_matrix, 'num_frames'):
            total_frames = time_series_matrix.num_frames
        if hasattr(time_series_matrix, 'object_ids'):
            detected_objects = len(time_series_matrix.object_ids)
        elif hasattr(time_series_matrix, 'num_objects'):
            detected_objects = time_series_matrix.num_objects
    
    # Fallback to other data sources
    if total_frames == "N/A":
        total_frames = processed_observation.get("total_frames", "N/A")
    if detected_objects == 0:
        detected_objects = processed_observation.get("detected_objects_count", 0)
        
    print(f"🎯 Dataset Overview:")
    print(f"   • Observation ID: {obs_id}")
    print(f"   • Total Frames: {total_frames}")
    print(f"   • Objects Detected: {detected_objects}")
    print(f"   • Frame Processing: ✓ Complete")
    
    # Data processing summary based on current pipeline configuration
    print(f"\n🔬 Data Processing Summary:")
    
    # Time series matrix processing
    if time_series_matrix:
        print(f"   • Time Series Matrix: ✓ Generated")
        if hasattr(time_series_matrix, 'trajectory_data') and time_series_matrix.trajectory_data is not None:
            if hasattr(time_series_matrix.trajectory_data, 'shape'):
                shape = time_series_matrix.trajectory_data.shape
                print(f"   • Trajectory Data Shape: {shape}")
    else:
        print(f"   • Time Series Matrix: ❌ Not generated")
    
    # Basic pipeline processing (may not be enabled based on process_observation parameters)
    if "segments" in processed_observation and processed_observation["segments"]:
        segments_count = len(processed_observation["segments"])
        print(f"   • Trajectory Segments: {segments_count}")
    else:
        print(f"   • Trajectory Segments: ❌ Not processed (extract_features=False)")
        
    if "bonds" in processed_observation and processed_observation["bonds"]:
        bonds_count = len(processed_observation["bonds"])
        print(f"   • Object Bonds: {bonds_count}")
    else:
        print(f"   • Object Bonds: ❌ Not processed (analyze_bonds=False)")
        
    if "classified_bonds" in processed_observation and processed_observation["classified_bonds"]:
        bond_types = processed_observation["classified_bonds"].get("bond_types", {})
        print(f"   • Bond Classification: {len(bond_types)} types identified")
    else:
        print(f"   • Bond Classification: ❌ Not processed (analyze_bonds=False)")
        
    # Facts and reasoning analysis (based on current process_single_observation)
    symbolic_facts = processed_observation.get("_symbolic_facts", {})
    if symbolic_facts and isinstance(symbolic_facts, dict):
        fact_categories = list(symbolic_facts.keys())
        total_facts = sum(len(facts) if isinstance(facts, (list, dict)) else 1 for facts in symbolic_facts.values())
        print(f"   • Symbolic Facts: ✓ {len(fact_categories)} categories, {total_facts} total facts")
        for category in fact_categories[:3]:  # Show first 3 categories
            print(f"     └─ {category}")
    else:
        print(f"   • Symbolic Facts: ❌ Not extracted or empty")
        
    # Note: base_rules are not generated in current process_single_observation
    print(f"   • Reasoning Rules: ❌ Not implemented in current pipeline")
        
    # Causal analysis results (this would come from causal_workspace processing)
    causal_edges = processed_observation.get("causal_edges", [])
    causal_matrix = processed_observation.get("causal_matrix")
    
    if causal_edges:
        print(f"   • Causal Analysis: ✓ Success")
        print(f"   • Causal Relationships: {len(causal_edges)} discovered")
    elif causal_matrix is not None:
        print(f"   • Causal Analysis: ✓ Matrix generated")
        print(f"   • Causal Relationships: Matrix-based analysis complete")
    elif "error" in processed_observation:
        print(f"   • Causal Analysis: ❌ Failed - {processed_observation.get('error', 'Unknown error')}")
    else:
        print(f"   • Causal Analysis: ⚠ No causal relationships discovered")
    
    # Processing status based on available results
    has_time_series = time_series_matrix is not None
    has_facts = bool(symbolic_facts)
    has_causal = bool(causal_edges) or causal_matrix is not None
    
    if has_time_series and has_facts:
        processing_status = "✅ Successful - Core pipeline complete"
    elif has_time_series:
        processing_status = "⚠ Partial - Time series generated, facts extraction incomplete"
    else:
        processing_status = "❌ Failed - Core processing incomplete"
        
    print(f"\n⚡ Processing Status: {processing_status}")
    
    print("="*80 + "\n")


def perform_cross_observation_analysis(all_results: Dict[str, Any], all_causal_graphs: Dict[str, List]) -> Dict[str, Any]:
    """Perform cross-observation analysis and build models.
    
    Args:
        all_results: Dictionary of results from all processed observations
        all_causal_graphs: Dictionary of causal graphs for all observations
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Performing cross-observation analysis...")
    
    analysis_results = {}
    
    try:
        # Build prediction models using all observations
        prediction_results = build_prediction_models(all_results, all_causal_graphs)
        logger.info("Prediction model building completed")
        analysis_results["prediction_results"] = prediction_results
    except NotImplementedError:
        logger.info("Prediction model building not yet implemented")
        analysis_results["prediction_results"] = {}
    
    try:
        # Validate pipeline results
        validation_results = validate_pipeline_results(all_results, all_causal_graphs)
        logger.info("Pipeline validation completed")
        analysis_results["validation_results"] = validation_results
    except NotImplementedError:
        logger.info("Pipeline validation not yet implemented")
        analysis_results["validation_results"] = {}
    
    return analysis_results


def log_final_summary(all_results: Dict[str, Any], all_causal_graphs: Dict[str, List]):
    """Log final pipeline completion summary.
    
    Args:
        all_results: Dictionary of results from all processed observations
        all_causal_graphs: Dictionary of causal graphs for all observations
    """
    total_scenarios = len(all_results)
    total_causal_edges = sum(len(edges) for edges in all_causal_graphs.values())
    
    logger.info("=== PIPELINE COMPLETION SUMMARY ===")
    logger.info(f"✅ Processed {total_scenarios} observations")
    logger.info(f"✅ Discovered {total_causal_edges} total causal relationships")
    # Note: safety_patterns variable needs to be defined somewhere else
    # logger.info(f"✅ Safety patterns identified: {len(safety_patterns)}")
    logger.info(f"✅ Pipeline framework ready for implementation")
    logger.info("Processing completed successfully!")


def main():
    """Complete usage of the video pipeline with causal analysis."""
    # Initialize pipeline and get observation IDs
    pipeline, observation_ids = initialize_pipeline()
    if pipeline is None:
        return
    
    # Display dataset overview
    display_dataset_overview(observation_ids)
    
    # Process all observations
    all_results = {}
    
    # Iterate through each observation ID
    for obs_id in observation_ids:
        # Process single observation
        processed_observation = process_single_observation_logic(pipeline, obs_id)
        all_results[obs_id] = processed_observation
        


    print("Program finished processing all observations.")

if __name__ == "__main__":
    main()