"""
Evaluation Metrics for CryptoGuard-LLM

Comprehensive metrics for evaluating fraud detection performance.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates evaluation metrics for fraud detection.
    
    Supports:
    - Standard classification metrics
    - Fraud-specific metrics (e.g., detection rate by fraud type)
    - Statistical significance tests
    - Confidence intervals
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.fraud_types = [
            'legitimate', 'rug_pull', 'phishing', 
            'ponzi_scheme', 'exchange_hack', 'ransomware'
        ]
        
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            
        return metrics
    
    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each class."""
        if class_names is None:
            class_names = [f'class_{i}' for i in range(self.num_classes)]
            
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        per_class = {}
        for i, name in enumerate(class_names):
            if str(i) in report:
                per_class[name] = {
                    'precision': report[str(i)]['precision'],
                    'recall': report[str(i)]['recall'],
                    'f1': report[str(i)]['f1-score'],
                    'support': int(report[str(i)]['support'])
                }
                
        return per_class
    
    def calculate_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals using bootstrap.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            confidence: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of metric names to (lower, upper) bounds
        """
        n = len(y_true)
        metrics_samples = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
            
            # Calculate metrics
            metrics_samples['accuracy'].append(accuracy_score(y_true_sample, y_pred_sample))
            metrics_samples['precision'].append(
                precision_score(y_true_sample, y_pred_sample, zero_division=0)
            )
            metrics_samples['recall'].append(
                recall_score(y_true_sample, y_pred_sample, zero_division=0)
            )
            metrics_samples['f1'].append(
                f1_score(y_true_sample, y_pred_sample, zero_division=0)
            )
            
        # Calculate confidence intervals
        alpha = 1 - confidence
        intervals = {}
        
        for metric, samples in metrics_samples.items():
            lower = np.percentile(samples, alpha / 2 * 100)
            upper = np.percentile(samples, (1 - alpha / 2) * 100)
            intervals[metric] = (lower, upper)
            
        return intervals
    
    def statistical_significance_test(
        self,
        metrics_model1: List[float],
        metrics_model2: List[float],
        test: str = 'paired_t'
    ) -> Dict[str, float]:
        """
        Test statistical significance between two models.
        
        Args:
            metrics_model1: Metric values from model 1 (e.g., from CV folds)
            metrics_model2: Metric values from model 2
            test: Type of test ('paired_t', 'wilcoxon', 'mcnemar')
            
        Returns:
            Dictionary with test statistic, p-value, and effect size
        """
        results = {}
        
        if test == 'paired_t':
            statistic, p_value = stats.ttest_rel(metrics_model1, metrics_model2)
            results['test'] = 'paired_t'
            results['statistic'] = statistic
            results['p_value'] = p_value
            
            # Cohen's d effect size
            diff = np.array(metrics_model1) - np.array(metrics_model2)
            results['cohens_d'] = np.mean(diff) / np.std(diff, ddof=1)
            
        elif test == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(metrics_model1, metrics_model2)
            results['test'] = 'wilcoxon'
            results['statistic'] = statistic
            results['p_value'] = p_value
            
        return results


class FraudDetectionMetrics:
    """
    Specialized metrics for fraud detection evaluation.
    
    Includes:
    - Detection rate by fraud amount
    - Time-to-detection metrics
    - Cost-sensitive evaluation
    """
    
    def __init__(
        self,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 10.0
    ):
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        
    def calculate_cost_sensitive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate cost-sensitive evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            amounts: Transaction amounts (optional)
            
        Returns:
            Dictionary of cost-sensitive metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic cost calculation
        total_cost = fp * self.fp_cost + fn * self.fn_cost
        
        metrics = {
            'total_cost': total_cost,
            'avg_cost_per_sample': total_cost / len(y_true),
            'fp_cost_total': fp * self.fp_cost,
            'fn_cost_total': fn * self.fn_cost
        }
        
        # Amount-based metrics (if amounts provided)
        if amounts is not None:
            fraud_mask = y_true == 1
            detected_mask = (y_true == 1) & (y_pred == 1)
            missed_mask = (y_true == 1) & (y_pred == 0)
            
            metrics['total_fraud_amount'] = float(amounts[fraud_mask].sum())
            metrics['detected_fraud_amount'] = float(amounts[detected_mask].sum())
            metrics['missed_fraud_amount'] = float(amounts[missed_mask].sum())
            
            if metrics['total_fraud_amount'] > 0:
                metrics['amount_detection_rate'] = (
                    metrics['detected_fraud_amount'] / metrics['total_fraud_amount']
                )
            else:
                metrics['amount_detection_rate'] = 0.0
                
        return metrics
    
    def calculate_temporal_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate time-based detection metrics.
        
        Args:
            predictions: List of prediction dicts with timestamps
            ground_truth: List of ground truth dicts with fraud start times
            
        Returns:
            Dictionary of temporal metrics
        """
        detection_times = []
        
        for gt in ground_truth:
            fraud_start = gt.get('fraud_start_time')
            if fraud_start is None:
                continue
                
            # Find first detection
            for pred in predictions:
                if pred.get('is_fraud') and pred.get('timestamp', 0) >= fraud_start:
                    detection_time = pred['timestamp'] - fraud_start
                    detection_times.append(detection_time)
                    break
                    
        if not detection_times:
            return {
                'avg_detection_time': float('inf'),
                'median_detection_time': float('inf'),
                'detection_rate': 0.0
            }
            
        return {
            'avg_detection_time': np.mean(detection_times),
            'median_detection_time': np.median(detection_times),
            'min_detection_time': np.min(detection_times),
            'max_detection_time': np.max(detection_times),
            'detection_rate': len(detection_times) / len(ground_truth)
        }
    
    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, float]]:
        """
        Calculate metrics at different decision thresholds.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            List of metric dictionaries for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
            
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            results.append({
                'threshold': threshold,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
            })
            
        return results
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find the optimal decision threshold.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_value = 0.0
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                value = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            if value > best_value:
                best_value = value
                best_threshold = threshold
                
        return best_threshold, best_value


if __name__ == '__main__':
    # Example usage
    print("MetricsCalculator - Evaluation metrics for fraud detection")
    
    # Generate sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.binomial(1, 0.1, n_samples)  # 10% fraud
    y_prob = np.random.beta(2, 5, n_samples)
    y_prob[y_true == 1] = np.random.beta(5, 2, y_true.sum())  # Higher probs for fraud
    y_pred = (y_prob > 0.5).astype(int)
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_prob)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
            
    # Confidence intervals
    ci = calculator.calculate_confidence_intervals(y_true, y_pred)
    print("\n95% Confidence Intervals:")
    for name, (lower, upper) in ci.items():
        print(f"  {name}: [{lower:.4f}, {upper:.4f}]")
