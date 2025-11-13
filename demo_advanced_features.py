"""
Demo: Advanced Feature Extraction and Temporal Bond Analysis

This script demonstrates the advanced features of the CauVid pipeline:
1. Feature vector extraction for each object in each frame
2. Temporal bond analysis using cosine similarity
3. Video segmentation based on bond breaks
4. Analysis of object behavior and key events
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import VideoPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_advanced_features():
    """Demonstrate advanced feature extraction and bond analysis."""
    
    # Setup
    TWO_PARTS_ROOT = "./two_parts"
    OUTPUT_DIR = "./pipeline_output"
    
    # Initialize pipeline with advanced features
    pipeline = VideoPipeline(
        TWO_PARTS_ROOT,
        OUTPUT_DIR,
        detector_type="circle",
        position_threshold=20.0,
        bond_threshold=0.8,  # Bond similarity threshold (τ)
        frame_height=480,
        frame_width=640
    )
    
    # Get first observation
    observation_ids = pipeline.frame_loader.get_observation_ids()
    obs_id = observation_ids[0]
    
    print(f"\n🎬 Processing observation: {obs_id}")
    print("=" * 60)
    
    # Process with all advanced features enabled
    results = pipeline.process_observation(
        obs_id,
        include_ground_truth=True,
        extract_features=True,
        analyze_bonds=True
    )
    
    # Extract components
    time_series_matrix = results['time_series_matrix']
    object_features = results['object_features']
    object_bonds = results['object_bonds']
    video_segments = results['video_segments']
    
    # 1. Feature Vector Analysis
    print("\n🧮 FEATURE VECTOR ANALYSIS")
    print("-" * 40)
    
    print(f"📊 Extracted features for {len(object_features)} objects")
    
    # Show detailed analysis for first object
    first_obj_id = list(object_features.keys())[0]
    first_obj_features = object_features[first_obj_id]
    
    print(f"\n🔍 Detailed analysis for {first_obj_id}:")
    print(f"   Frames tracked: {len(first_obj_features)}")
    
    # Feature statistics across all frames for this object
    feature_arrays = {
        'velocity_direction': [],
        'speed': [],
        'contact_pattern': [],
        'support_pattern': [],
        'kinetic_energy': [],
        'potential_energy': []
    }
    
    for frame_idx, features in first_obj_features.items():
        feature_arrays['velocity_direction'].append(features.mean_velocity_direction)
        feature_arrays['speed'].append(features.mean_speed)
        feature_arrays['contact_pattern'].append(features.contact_pattern)
        feature_arrays['support_pattern'].append(features.support_pattern)
        feature_arrays['kinetic_energy'].append(features.kinetic_energy)
        feature_arrays['potential_energy'].append(features.potential_energy)
    
    print("   Feature statistics:")
    for feat_name, values in feature_arrays.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"     {feat_name}: μ={mean_val:.3f} σ={std_val:.3f} range=[{min_val:.3f}, {max_val:.3f}]")
    
    # 2. Bond Analysis
    print(f"\n🔗 TEMPORAL BOND ANALYSIS")
    print("-" * 40)
    
    total_bonds = sum(len(bonds) for bonds in object_bonds.values())
    broken_bonds = sum(1 for bonds in object_bonds.values() for bond in bonds if not bond.bond_exists)
    key_events = sum(1 for bonds in object_bonds.values() for bond in bonds if bond.key_event_detected)
    
    print(f"📈 Bond Statistics:")
    print(f"   Total frame transitions: {total_bonds}")
    print(f"   Strong bonds maintained: {total_bonds - broken_bonds} ({(total_bonds-broken_bonds)/total_bonds*100:.1f}%)")
    print(f"   Bonds broken: {broken_bonds} ({broken_bonds/total_bonds*100:.1f}%)")
    print(f"   Key events detected: {key_events}")
    
    # Bond strength distribution
    all_similarities = [bond.similarity for bonds in object_bonds.values() for bond in bonds]
    print(f"\n📊 Bond Strength Distribution:")
    print(f"   Mean similarity: {np.mean(all_similarities):.3f}")
    print(f"   Std similarity: {np.std(all_similarities):.3f}")
    print(f"   Min similarity: {np.min(all_similarities):.3f}")
    print(f"   Max similarity: {np.max(all_similarities):.3f}")
    
    # Show bond patterns for first object
    first_obj_bonds = object_bonds[first_obj_id]
    print(f"\n🔍 Bond pattern for {first_obj_id}:")
    
    bond_break_frames = [bond.frame_t1 for bond in first_obj_bonds if not bond.bond_exists]
    key_event_frames = [bond.frame_t1 for bond in first_obj_bonds if bond.key_event_detected]
    
    print(f"   Bond breaks at frames: {bond_break_frames}")
    print(f"   Key events at frames: {key_event_frames}")
    
    # Show sample bonds
    print("   Sample transitions:")
    for i, bond in enumerate(first_obj_bonds[:5]):
        status = "STRONG" if bond.bond_exists else "BREAK"
        key_event = " ⚡KEY_EVENT" if bond.key_event_detected else ""
        print(f"     {bond.frame_t}→{bond.frame_t1}: {bond.similarity:.3f} [{status}]{key_event}")
    
    # 3. Video Segmentation
    print(f"\n🎞️  VIDEO SEGMENTATION")
    print("-" * 40)
    
    print(f"📹 Segmentation Results:")
    print(f"   Total segments: {len(video_segments)}")
    
    total_frames = sum(seg.end_frame - seg.start_frame + 1 for seg in video_segments)
    print(f"   Total frames: {total_frames}")
    
    # Segment analysis
    segment_lengths = [seg.end_frame - seg.start_frame + 1 for seg in video_segments]
    print(f"   Average segment length: {np.mean(segment_lengths):.1f} frames")
    print(f"   Segment length range: [{np.min(segment_lengths)}, {np.max(segment_lengths)}]")
    
    # Break reason analysis
    break_reasons = {}
    for seg in video_segments:
        reason = seg.break_reason
        break_reasons[reason] = break_reasons.get(reason, 0) + 1
    
    print(f"\n📊 Segment break reasons:")
    for reason, count in break_reasons.items():
        print(f"   {reason}: {count} segments")
    
    # Show detailed segments
    print(f"\n🔍 Detailed segment breakdown:")
    for segment in video_segments:
        duration = segment.end_frame - segment.start_frame + 1
        print(f"   {segment.segment_id}: frames {segment.start_frame}-{segment.end_frame}")
        print(f"     Duration: {duration} frames")
        print(f"     Avg bond strength: {segment.avg_bond_strength:.3f}")
        print(f"     Break reason: {segment.break_reason}")
        print(f"     Objects: {len(segment.object_ids)}")
    
    # 4. Trajectory Analysis with Features
    print(f"\n🎯 TRAJECTORY + FEATURE ANALYSIS")
    print("-" * 40)
    
    # Get trajectory
    trajectory = time_series_matrix.get_object_trajectory(first_obj_id)
    
    print(f"🎯 Enhanced trajectory analysis for {first_obj_id}:")
    print(f"   Position displacement: {np.sqrt((trajectory['position_x'][-1] - trajectory['position_x'][0])**2 + (trajectory['position_y'][-1] - trajectory['position_y'][0])**2):.2f} px")
    print(f"   Mean velocity: ({np.mean(trajectory['velocity_x']):.2f}, {np.mean(trajectory['velocity_y']):.2f}) px/frame")
    
    # Correlate with features
    speeds = [first_obj_features[i].mean_speed for i in range(len(first_obj_features))]
    kinetic_energies = [first_obj_features[i].kinetic_energy for i in range(len(first_obj_features))]
    
    print(f"   Feature-derived mean speed: {np.mean(speeds):.3f}")
    print(f"   Mean kinetic energy: {np.mean(kinetic_energies):.3f}")
    print(f"   Energy variance: {np.var(kinetic_energies):.6f} (stability indicator)")
    
    # 5. Output Summary
    print(f"\n📁 OUTPUT FILES")
    print("-" * 40)
    
    output_files = [
        f"{obs_id}_processed.json",
        f"{obs_id}_timeseries.json", 
        f"{obs_id}_timeseries.npz",
        f"{obs_id}_features.json",
        f"{obs_id}_bonds.json",
        f"{obs_id}_segments.json"
    ]
    
    for filename in output_files:
        filepath = Path(OUTPUT_DIR) / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {filename} (missing)")
    
    print("\n🎉 Advanced feature analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("CauVid Advanced Features Demo")
    print("=============================")
    
    demonstrate_advanced_features()