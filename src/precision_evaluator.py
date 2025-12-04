"""
Precision evaluation utilities for object detection.
"""

import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import euclidean
import logging

from data_models import DetectedObject, PrecisionMetrics

logger = logging.getLogger(__name__)


class PrecisionEvaluator:
    """Evaluates precision and recall of object detection against ground truth."""
    
    def __init__(self, position_threshold: float = 20.0):
        self.position_threshold = position_threshold
    
    def evaluate_frame(self, detected_objects: List[DetectedObject], 
                      ground_truth_objects: List[DetectedObject]) -> PrecisionMetrics:
        """Evaluate detection performance for a single frame."""
        
        if not ground_truth_objects:
            # No ground truth objects
            if not detected_objects:
                return PrecisionMetrics(1.0, 1.0, 1.0, 0.0, 0, 0, 0)
            else:
                return PrecisionMetrics(0.0, 0.0, 0.0, 0.0, 0, len(detected_objects), 0)
        
        if not detected_objects:
            # No detections but ground truth exists
            return PrecisionMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, len(ground_truth_objects))
        
        # Match detected objects to ground truth
        matches = self._match_objects(detected_objects, ground_truth_objects)
        
        true_positives = len(matches)
        false_positives = len(detected_objects) - true_positives
        false_negatives = len(ground_truth_objects) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate mean position error for matched objects
        position_errors = []
        for detected_idx, gt_idx in matches:
            detected_center = detected_objects[detected_idx].center
            gt_center = ground_truth_objects[gt_idx].center
            error = euclidean(detected_center, gt_center)
            position_errors.append(error)
        
        mean_position_error = np.mean(position_errors) if position_errors else 0.0
        
        return PrecisionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_position_error=mean_position_error,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def _match_objects(self, detected_objects: List[DetectedObject], 
                      ground_truth_objects: List[DetectedObject]) -> List[tuple]:
        """Match detected objects to ground truth objects based on position."""
        
        matches = []
        used_gt_indices = set()
        
        for det_idx, detected_obj in enumerate(detected_objects):
            best_match_idx = None
            best_distance = float('inf')
            
            for gt_idx, gt_obj in enumerate(ground_truth_objects):
                if gt_idx in used_gt_indices:
                    continue
                
                # Calculate distance between centers
                distance = euclidean(detected_obj.center, gt_obj.center)
                
                if distance <= self.position_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = gt_idx
            
            if best_match_idx is not None:
                matches.append((det_idx, best_match_idx))
                used_gt_indices.add(best_match_idx)
        
        return matches
    
    def evaluate_sequence(self, frame_evaluations: List[PrecisionMetrics]) -> PrecisionMetrics:
        """Evaluate overall performance across a sequence of frames."""
        
        if not frame_evaluations:
            return PrecisionMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
        
        # Aggregate metrics
        total_tp = sum(metrics.true_positives for metrics in frame_evaluations)
        total_fp = sum(metrics.false_positives for metrics in frame_evaluations)
        total_fn = sum(metrics.false_negatives for metrics in frame_evaluations)
        
        # Calculate overall precision, recall, F1
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Calculate mean position error (weighted by number of matches)
        total_weighted_error = 0.0
        total_matches = 0
        
        for metrics in frame_evaluations:
            if metrics.true_positives > 0:
                total_weighted_error += metrics.mean_position_error * metrics.true_positives
                total_matches += metrics.true_positives
        
        overall_mean_error = total_weighted_error / total_matches if total_matches > 0 else 0.0
        
        return PrecisionMetrics(
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            mean_position_error=overall_mean_error,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn
        )
    
    def get_detailed_report(self, metrics: PrecisionMetrics) -> str:
        """Generate a detailed text report of the evaluation metrics."""
        
        report = f"""
Object Detection Evaluation Report
==================================
Precision: {metrics.precision:.3f}
Recall: {metrics.recall:.3f}
F1-Score: {metrics.f1_score:.3f}
Mean Position Error: {metrics.mean_position_error:.2f} pixels

Confusion Matrix:
- True Positives: {metrics.true_positives}
- False Positives: {metrics.false_positives}
- False Negatives: {metrics.false_negatives}
"""
        return report