# ============================================================
#  SmartAir-Guardian — Metrics and Visualization Utilities
#  model/utils/metrics.py
#
#  Functions:
#    - plot_training_history(history, save_path)
#    - plot_confusion_matrix(y_true, y_pred, class_names, save_path)
#    - print_classification_report(y_true, y_pred, class_names, save_path)
#    - compare_models_table(results, save_path)
# ============================================================

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/metrics from Keras history.
    
    Args:
        history: Keras history object from model.fit()
        save_path: Path to save the figure (if None, display only)
    """
    if not hasattr(history, 'history'):
        print("[WARN] Invalid history object")
        return
    
    hist = history.history
    
    # Determine available metrics
    available_keys = list(hist.keys())
    print(f"[METRICS] Available keys: {available_keys}")
    
    # Filter to get loss and validation loss
    loss_keys = [k for k in available_keys if 'loss' in k.lower() and 'val_' not in k]
    val_loss_keys = [k for k in available_keys if 'val_' in k and 'loss' in k.lower()]
    
    if not loss_keys or not val_loss_keys:
        print("[WARN] No loss keys found in history")
        return
    
    loss_key = loss_keys[0]
    val_loss_key = val_loss_keys[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(hist[loss_key], label='Training Loss', linewidth=2)
    axes[0].plot(hist[val_loss_key], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Metrics (accuracy or others)
    metric_keys = [k for k in available_keys if 'acc' in k.lower() and 'val_' not in k]
    if metric_keys:
        met_key = metric_keys[0]
        val_met_keys = [k for k in available_keys if 'val_' in k and 'acc' in k.lower()]
        if val_met_keys:
            val_met_key = val_met_keys[0]
            axes[1].plot(hist[met_key], label='Training Accuracy', linewidth=2)
            axes[1].plot(hist[val_met_key], label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Model Accuracy Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[METRICS] Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save confusion matrix heatmap.
    
    Args:
        y_true: True class labels (1D array)
        y_pred: Predicted class labels (1D array)
        class_names: List of class name strings
        save_path: Path to save the figure (if None, display only)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[METRICS] Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Generate and optionally save detailed classification report.
    
    Args:
        y_true: True class labels (1D array)
        y_pred: Predicted class labels (1D array)
        class_names: List of class name strings
        save_path: Path to save the report as text (if None, print only)
    
    Returns:
        dict: Dictionary with metrics (accuracy, f1, precision, recall)
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    output = f"""
{'='*60}
CLASSIFICATION REPORT
{'='*60}

{report}

{'='*60}
SUMMARY METRICS
{'='*60}
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}
{'='*60}
"""
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(output)
        print(f"[METRICS] Saved classification report to {save_path}")
    else:
        print(output)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compare_models_table(results, save_path=None):
    """
    Build and display comparison table of multiple models.
    
    Args:
        results: dict with model names as keys, metrics dicts as values
                 Each metrics dict should have: accuracy, f1, precision, recall
        save_path: Path to save the table as text (if None, print only)
    """
    # Sort by accuracy descending
    sorted_results = dict(
        sorted(results.items(),
               key=lambda x: x[1].get('accuracy', 0),
               reverse=True)
    )
    
    output = f"""
{'='*80}
MODEL COMPARISON TABLE
{'='*80}

{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}
{'-'*80}
"""
    
    for model_name, metrics in sorted_results.items():
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        output += f"{model_name:<30} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}\n"
    
    output += f"{'='*80}\n"
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(output)
        print(f"[METRICS] Saved model comparison table to {save_path}")
    else:
        print(output)
