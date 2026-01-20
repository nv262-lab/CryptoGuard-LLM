"""
Visualization Utilities for CryptoGuard-LLM

Plotting functions for model evaluation and analysis.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for classes
        normalize: Whether to normalize values
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Legitimate', 'Fraud']
        
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
        
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
        
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
               label=f'Baseline (prevalence = {baseline:.3f})')
    
    ax.fill_between(recall, precision, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PR curve to {save_path}")
        
    return fig


class Visualizer:
    """
    Visualization class for CryptoGuard-LLM.
    
    Provides methods for:
    - Model performance visualization
    - Training progress plots
    - Feature importance visualization
    - Transaction graph visualization
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        plt.style.use(style)
        self.figures = {}
        
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = 'Training History',
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training metrics over epochs."""
        n_metrics = len(history)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, (metric, values) in zip(axes, history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Training')
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        self.figures['training_history'] = fig
        return fig
    
    def plot_model_comparison(
        self,
        models: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
        title: str = 'Model Performance Comparison',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of multiple models."""
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, (model_name, model_metrics) in enumerate(models.items()):
            values = [model_metrics.get(m, 0) for m in metrics]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
                
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        self.figures['model_comparison'] = fig
        return fig
    
    def plot_fraud_distribution(
        self,
        fraud_types: List[str],
        counts: List[int],
        title: str = 'Distribution of Fraud Types',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of fraud types."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(fraud_types)))
        ax1.pie(counts, labels=fraud_types, colors=colors, autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Fraud Type Distribution')
        
        # Bar chart
        bars = ax2.barh(fraud_types, counts, color=colors)
        ax2.set_xlabel('Count')
        ax2.set_title('Fraud Counts by Type')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax2.annotate(f'{count:,}',
                        xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        ha='left', va='center', fontsize=9)
            
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        self.figures['fraud_distribution'] = fig
        return fig
    
    def plot_temporal_analysis(
        self,
        timestamps: List,
        detections: List[int],
        false_positives: List[int],
        response_times: List[float],
        title: str = 'Temporal Performance Analysis',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot temporal analysis of detection performance."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Detections over time
        axes[0].plot(timestamps, detections, 'b-', linewidth=2, marker='o')
        axes[0].set_ylabel('Threats Detected')
        axes[0].set_title('Threat Detection Volume')
        axes[0].fill_between(timestamps, detections, alpha=0.3)
        
        # False positives over time
        axes[1].plot(timestamps, false_positives, 'r-', linewidth=2, marker='s')
        axes[1].set_ylabel('False Positives')
        axes[1].set_title('False Positive Rate')
        axes[1].fill_between(timestamps, false_positives, alpha=0.3, color='red')
        
        # Response time over time
        axes[2].plot(timestamps, response_times, 'g-', linewidth=2, marker='^')
        axes[2].set_ylabel('Response Time (min)')
        axes[2].set_xlabel('Time Period')
        axes[2].set_title('Average Response Time')
        axes[2].fill_between(timestamps, response_times, alpha=0.3, color='green')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        self.figures['temporal_analysis'] = fig
        return fig
    
    def save_all_figures(self, output_dir: str):
        """Save all generated figures."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            path = os.path.join(output_dir, f'{name}.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved {name} to {path}")


if __name__ == '__main__':
    # Example usage
    print("Visualization utilities for CryptoGuard-LLM")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, 1000)
    y_prob = np.random.beta(2, 5, 1000)
    y_prob[y_true == 1] = np.random.beta(5, 2, y_true.sum())
    y_pred = (y_prob > 0.5).astype(int)
    
    # Plot examples
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)
    plot_precision_recall_curve(y_true, y_prob)
    
    plt.show()
