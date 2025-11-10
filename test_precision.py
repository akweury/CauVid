"""
Test script for precision calculation in the CauVid pipeline

This script demonstrates the precision calculation features and provides
detailed analysis of detection accuracy against ground truth.
"""

import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from video_pipeline import VideoPipeline, PrecisionEvaluator, ObjectDetector, VideoFrameLoader
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_precision_calculation():
    """Test precision calculation on a single observation."""
    
    TWO_PARTS_ROOT = "./two_parts"
    OUTPUT_DIR = "./precision_test_output"
    
    logger.info("Testing precision calculation functionality")
    
    try:
        # Initialize pipeline with precision evaluation
        pipeline = VideoPipeline(
            TWO_PARTS_ROOT, 
            OUTPUT_DIR, 
            detector_type="circle",
            position_threshold=20.0  # 20 pixel tolerance
        )
        
        # Get first observation
        observation_ids = pipeline.frame_loader.get_observation_ids()
        if not observation_ids:
            logger.error("No observations found!")
            return
        
        test_obs = observation_ids[0]
        logger.info(f"Testing precision on: {test_obs}")
        
        # Process with precision calculation
        matrix = pipeline.process_observation(test_obs, include_ground_truth=True)
        
        # Display detailed results
        logger.info("\n" + "="*60)
        logger.info("DETAILED PRECISION ANALYSIS")
        logger.info("="*60)
        
        # Overall metrics
        if matrix.overall_metrics:
            metrics = matrix.overall_metrics
            logger.info(f"\nOVERALL PERFORMANCE:")
            logger.info(f"  📊 Precision: {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
            logger.info(f"  📊 Recall: {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
            logger.info(f"  📊 F1-Score: {metrics.f1_score:.4f}")
            logger.info(f"  📏 Mean Position Error: {metrics.mean_position_error:.2f} pixels")
            logger.info(f"  ✅ True Positives: {metrics.true_positives}")
            logger.info(f"  ❌ False Positives: {metrics.false_positives}")
            logger.info(f"  ⚠️  False Negatives: {metrics.false_negatives}")
        
        # Frame-by-frame analysis for first 5 frames
        logger.info(f"\nFRAME-BY-FRAME ANALYSIS (First 5 frames):")
        logger.info("-" * 50)
        
        for frame_idx in range(min(5, len(matrix.frames_data))):
            frame_data = matrix.frames_data[frame_idx]
            if frame_data.precision_metrics:
                m = frame_data.precision_metrics
                logger.info(f"Frame {frame_idx:2d}: "
                          f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1_score:.3f} "
                          f"Err={m.mean_position_error:5.1f}px "
                          f"(TP:{m.true_positives} FP:{m.false_positives} FN:{m.false_negatives})")
        
        # Analyze detection quality
        analyze_detection_quality(matrix)
        
        return matrix
        
    except Exception as e:
        logger.error(f"Error in precision test: {e}")
        raise


def analyze_detection_quality(matrix):
    """Perform detailed analysis of detection quality."""
    
    logger.info(f"\nDETECTION QUALITY ANALYSIS:")
    logger.info("-" * 40)
    
    position_errors = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for frame_data in matrix.frames_data.values():
        if frame_data.precision_metrics:
            m = frame_data.precision_metrics
            if m.mean_position_error > 0:
                position_errors.append(m.mean_position_error)
            precision_scores.append(m.precision)
            recall_scores.append(m.recall)
            f1_scores.append(m.f1_score)
    
    if position_errors:
        logger.info(f"Position Error Statistics:")
        logger.info(f"  Mean: {np.mean(position_errors):.2f} ± {np.std(position_errors):.2f} pixels")
        logger.info(f"  Min/Max: {np.min(position_errors):.2f} / {np.max(position_errors):.2f} pixels")
        logger.info(f"  Median: {np.median(position_errors):.2f} pixels")
    
    if precision_scores:
        logger.info(f"Precision Statistics:")
        logger.info(f"  Mean: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
        logger.info(f"  Min/Max: {np.min(precision_scores):.4f} / {np.max(precision_scores):.4f}")
    
    if recall_scores:
        logger.info(f"Recall Statistics:")
        logger.info(f"  Mean: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
        logger.info(f"  Min/Max: {np.min(recall_scores):.4f} / {np.max(recall_scores):.4f}")


def compare_different_thresholds():
    """Test precision with different position thresholds."""
    
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("="*60)
    
    TWO_PARTS_ROOT = "./two_parts"
    
    # Test different thresholds
    thresholds = [5, 10, 15, 20, 25, 30]
    results = {}
    
    # Get test observation
    frame_loader = VideoFrameLoader(TWO_PARTS_ROOT)
    observation_ids = frame_loader.get_observation_ids()
    test_obs = observation_ids[0]
    
    # Load test data
    frames = frame_loader.load_frames(test_obs)
    detector = ObjectDetector(model_type="circle")
    
    for threshold in thresholds:
        logger.info(f"Testing threshold: {threshold} pixels")
        
        evaluator = PrecisionEvaluator(position_threshold=threshold)
        
        # Test on first 10 frames
        frame_metrics = []
        for frame_idx in range(min(10, len(frames))):
            # Get detections
            detected = detector.detect_objects(frames[frame_idx], frame_idx)
            
            # Get ground truth
            gt_objects = frame_loader.get_ground_truth_objects(test_obs, frame_idx)
            
            # Calculate metrics
            metrics = evaluator.calculate_frame_precision(detected, gt_objects)
            frame_metrics.append(metrics)
        
        # Calculate overall metrics for this threshold
        overall = evaluator.calculate_overall_metrics(frame_metrics)
        results[threshold] = overall
        
        logger.info(f"  Threshold {threshold:2d}px: "
                  f"P={overall.precision:.3f} R={overall.recall:.3f} F1={overall.f1_score:.3f} "
                  f"Err={overall.mean_position_error:.1f}px")
    
    # Find best threshold
    best_threshold = max(results.keys(), key=lambda t: results[t].f1_score)
    logger.info(f"\nBest threshold: {best_threshold} pixels (F1={results[best_threshold].f1_score:.3f})")
    
    return results


def create_precision_visualization(matrix):
    """Create visualizations of precision metrics."""
    
    logger.info("Creating precision visualizations...")
    
    # Extract data for plotting
    frame_indices = []
    precisions = []
    recalls = []
    f1_scores = []
    position_errors = []
    
    for frame_idx in sorted(matrix.frames_data.keys()):
        frame_data = matrix.frames_data[frame_idx]
        if frame_data.precision_metrics:
            m = frame_data.precision_metrics
            frame_indices.append(frame_idx)
            precisions.append(m.precision)
            recalls.append(m.recall)
            f1_scores.append(m.f1_score)
            if m.mean_position_error > 0:
                position_errors.append(m.mean_position_error)
            else:
                position_errors.append(0)
    
    if not frame_indices:
        logger.warning("No precision data to visualize")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detection Precision Analysis', fontsize=16)
    
    # 1. Precision, Recall, F1 over time
    ax1 = axes[0, 0]
    ax1.plot(frame_indices, precisions, 'b-o', label='Precision', markersize=3)
    ax1.plot(frame_indices, recalls, 'r-s', label='Recall', markersize=3)
    ax1.plot(frame_indices, f1_scores, 'g-^', label='F1-Score', markersize=3)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision Metrics Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 2. Position error over time
    ax2 = axes[0, 1]
    ax2.plot(frame_indices, position_errors, 'purple', marker='o', markersize=3)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Position Error (pixels)')
    ax2.set_title('Position Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    if position_errors:
        mean_error = np.mean([e for e in position_errors if e > 0])
        ax2.axhline(mean_error, color='red', linestyle='--', 
                   label=f'Mean: {mean_error:.2f}px')
        ax2.legend()
    
    # 3. Precision distribution
    ax3 = axes[1, 0]
    ax3.hist(precisions, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Precision Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Precision Score Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Overall metrics summary
    ax4 = axes[1, 1]
    if matrix.overall_metrics:
        metrics = matrix.overall_metrics
        categories = ['Precision', 'Recall', 'F1-Score']
        values = [metrics.precision, metrics.recall, metrics.f1_score]
        colors = ['blue', 'red', 'green']
        
        bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Score')
        ax4.set_title('Overall Performance Metrics')
        ax4.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "precision_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved precision analysis to {output_file}")
    
    plt.show()


def main():
    """Main function to run precision tests."""
    
    print("CauVid Precision Calculation Testing")
    print("=" * 50)
    
    try:
        # Test 1: Basic precision calculation
        print("\n1. Testing precision calculation...")
        matrix = test_precision_calculation()
        
        # Test 2: Threshold sensitivity
        print("\n2. Testing threshold sensitivity...")
        threshold_results = compare_different_thresholds()
        
        # Test 3: Create visualizations
        print("\n3. Creating visualizations...")
        create_precision_visualization(matrix)
        
        print("\nPrecision testing completed!")
        print("Check the generated output files and visualizations.")
        
    except Exception as e:
        logger.error(f"Precision testing failed: {e}")
        raise


if __name__ == "__main__":
    main()