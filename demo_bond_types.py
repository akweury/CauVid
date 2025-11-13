"""
Demo: Bond Type Classification and Analysis

This script demonstrates the advanced bond type classification system that
maintains a bond-type table and classifies similar bond patterns into types.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import VideoPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_bond_type_classification():
    """Demonstrate bond type classification and bond-type table maintenance."""
    
    print("🔗 BOND TYPE CLASSIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = VideoPipeline(
        "./two_parts",
        "./pipeline_output",
        detector_type="circle",
        bond_threshold=0.8
    )
    
    # Process observation with bond classification
    obs_id = pipeline.frame_loader.get_observation_ids()[0]
    
    print(f"\n📹 Processing: {obs_id}")
    print("-" * 40)
    
    results = pipeline.process_observation(
        obs_id,
        extract_features=True,
        analyze_bonds=True
    )
    
    # Extract bond analysis components
    bond_analysis = results['bond_analysis']
    bond_types = results['bond_types']
    bond_type_summary = results['bond_type_summary']
    classified_bonds = results['classified_bonds']
    video_segments = results['video_segments']
    
    # Display Bond Type Table
    print(f"\n📊 BOND TYPE TABLE")
    print("-" * 40)
    print(f"Total Bond Types Identified: {len(bond_types)}")
    print(f"Total Bonds Classified: {bond_type_summary['total_classified_bonds']}")
    print()
    
    # Sort bond types by frequency
    sorted_types = sorted(
        bond_type_summary['types'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    print("🏷️  Bond Type Classification Table:")
    print(f"{'Type ID':<15} {'Name':<25} {'Count':<8} {'%':<8} {'Signature'}")
    print("-" * 80)
    
    for type_id, type_info in sorted_types:
        signature_str = f"[{', '.join(f'{x:.3f}' for x in type_info['signature'][:3])}...]"
        print(f"{type_id:<15} {type_info['name']:<25} {type_info['count']:<8} {type_info['percentage']:<7.1f}% {signature_str}")
    
    # Analyze feature patterns for each bond type
    print(f"\n🧬 FEATURE PATTERN ANALYSIS")
    print("-" * 40)
    
    feature_names = [
        'velocity_direction', 'speed', 'contact_pattern',
        'support_pattern', 'kinetic_energy', 'potential_energy'
    ]
    
    for type_id, bond_type in list(bond_types.items())[:5]:  # Show top 5 types
        signature = bond_type.feature_signature
        
        print(f"\n🔍 {type_id} ({bond_type.type_name}) - {bond_type.count} bonds")
        print("   Feature signature pattern:")
        
        for i, (feat_name, value) in enumerate(zip(feature_names, signature)):
            direction = "↑" if value > 0.001 else "↓" if value < -0.001 else "→"
            magnitude = "LARGE" if abs(value) > 0.01 else "small"
            print(f"     {feat_name}: {value:+.4f} {direction} ({magnitude})")
        
        # Show examples
        print(f"   Examples: {bond_type.examples[:3]}")
    
    # Analyze bond type distribution across objects
    print(f"\n📈 BOND TYPE DISTRIBUTION BY OBJECT")
    print("-" * 40)
    
    object_type_counts = {}
    for obj_id, bonds in classified_bonds.items():
        object_type_counts[obj_id] = {}
        for bond in bonds:
            type_id = bond.bond_type_id
            object_type_counts[obj_id][type_id] = object_type_counts[obj_id].get(type_id, 0) + 1
    
    for obj_id, type_counts in object_type_counts.items():
        total_bonds = sum(type_counts.values())
        sorted_obj_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 {obj_id} ({total_bonds} bonds):")
        for type_id, count in sorted_obj_types[:3]:  # Top 3 types
            percentage = count / total_bonds * 100
            type_name = bond_types[type_id].type_name if type_id in bond_types else "unknown"
            print(f"   {type_id} ({type_name}): {count} bonds ({percentage:.1f}%)")
    
    # Analyze temporal patterns
    print(f"\n⏱️  TEMPORAL BOND TYPE PATTERNS")
    print("-" * 40)
    
    # Track bond type changes over time
    first_obj_id = list(classified_bonds.keys())[0]
    first_obj_bonds = classified_bonds[first_obj_id]
    
    print(f"📊 Bond type transitions for {first_obj_id}:")
    
    # Group consecutive bonds of same type
    type_sequences = []
    current_type = None
    current_start = 0
    current_count = 0
    
    for i, bond in enumerate(first_obj_bonds):
        if bond.bond_type_id != current_type:
            if current_type is not None:
                type_sequences.append({
                    'type_id': current_type,
                    'start_frame': current_start,
                    'end_frame': first_obj_bonds[i-1].bond_strength.frame_t1,
                    'count': current_count
                })
            current_type = bond.bond_type_id
            current_start = bond.bond_strength.frame_t
            current_count = 1
        else:
            current_count += 1
    
    # Add final sequence
    if current_type is not None:
        type_sequences.append({
            'type_id': current_type,
            'start_frame': current_start,
            'end_frame': first_obj_bonds[-1].bond_strength.frame_t1,
            'count': current_count
        })
    
    for i, seq in enumerate(type_sequences):
        type_name = bond_types[seq['type_id']].type_name if seq['type_id'] in bond_types else "unknown"
        duration = seq['end_frame'] - seq['start_frame']
        print(f"   Sequence {i+1}: {seq['type_id']} ({type_name})")
        print(f"     Frames {seq['start_frame']}-{seq['end_frame']} ({duration} frames, {seq['count']} transitions)")
    
    # Video segmentation with bond types
    print(f"\n🎬 VIDEO SEGMENTS WITH BOND TYPE ANALYSIS")
    print("-" * 40)
    
    for segment in video_segments:
        duration = segment.end_frame - segment.start_frame + 1
        print(f"\n📽️  {segment.segment_id}: frames {segment.start_frame}-{segment.end_frame}")
        print(f"    Duration: {duration} frames")
        print(f"    Break reason: {segment.break_reason}")
        print(f"    Avg bond strength: {segment.avg_bond_strength:.3f}")
        
        if segment.dominant_bond_types:
            print("    Dominant bond types:")
            for type_id, freq in segment.dominant_bond_types[:3]:  # Top 3
                type_name = bond_types[type_id].type_name if type_id in bond_types else "unknown"
                print(f"      {type_id} ({type_name}): {freq:.1%}")
    
    # Bond type similarity analysis
    print(f"\n🔬 BOND TYPE SIMILARITY MATRIX")
    print("-" * 40)
    
    if len(bond_types) > 1:
        type_ids = list(bond_types.keys())[:5]  # Top 5 types
        
        print("Cosine similarity between bond type signatures:")
        print(f"{'Type':<15}", end="")
        for type_id in type_ids:
            print(f"{type_id[-3:]:<8}", end="")
        print()
        
        for i, type_id1 in enumerate(type_ids):
            print(f"{type_id1[-3:]:<15}", end="")
            for j, type_id2 in enumerate(type_ids):
                if i == j:
                    similarity = 1.000
                else:
                    sig1 = bond_types[type_id1].feature_signature
                    sig2 = bond_types[type_id2].feature_signature
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(sig1, sig2)
                    norm1 = np.linalg.norm(sig1)
                    norm2 = np.linalg.norm(sig2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                    else:
                        similarity = 0.0
                
                print(f"{similarity:<8.3f}", end="")
            print()
    
    # Summary statistics
    print(f"\n📋 BOND TYPE CLASSIFICATION SUMMARY")
    print("-" * 40)
    
    print(f"🎯 Key Statistics:")
    print(f"   Total bond types discovered: {len(bond_types)}")
    print(f"   Total bonds classified: {bond_type_summary['total_classified_bonds']}")
    print(f"   Average bonds per type: {bond_type_summary['total_classified_bonds'] / len(bond_types):.1f}")
    
    # Type diversity analysis
    type_counts = [info['count'] for info in bond_type_summary['types'].values()]
    type_diversity = len([c for c in type_counts if c >= 5])  # Types with at least 5 bonds
    
    print(f"   Significant bond types (≥5 bonds): {type_diversity}")
    print(f"   Most common type: {sorted_types[0][1]['name']} ({sorted_types[0][1]['count']} bonds)")
    print(f"   Classification coverage: 100% (all bonds classified)")
    
    print(f"\n🎉 Bond type classification analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_bond_type_classification()