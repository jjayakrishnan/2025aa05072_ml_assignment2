"""
Model evaluation utilities
Calculate all 6 required metrics and generate visualizations
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all 6 required evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # 1. Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    # 2. AUC Score (requires probabilities)
    if y_pred_proba is not None:
        # Handle binary classification
        if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
            metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['AUC Score'] = None
    
    # 3. Precision
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    # 4. Recall
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # 5. F1 Score
    metrics['F1 Score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 6. Matthews Correlation Coefficient
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Pretty print metrics"""
    print(f"\n{'='*50}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name:20s}: {value:.4f}")
        else:
            print(f"{metric_name:20s}: N/A")
    print(f"{'='*50}\n")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix as heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the plot (optional)
    
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], 
                yticklabels=['No', 'Yes'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    return cm


def get_classification_report(y_true, y_pred):
    """
    Get classification report as dictionary
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Classification report as dictionary
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=['No', 'Yes'],
                                   output_dict=True)
    return report


def compare_models(results_dict):
    """
    Compare multiple models and create comparison table
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
    
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results_dict).T
    
    # Sort by F1 Score (or any other metric)
    df = df.sort_values('F1 Score', ascending=False)
    
    return df


def plot_model_comparison(results_df, save_path=None):
    """
    Plot model comparison chart
    
    Args:
        results_df: DataFrame with model comparison
        save_path: Path to save the plot (optional)
    """
    # Select metrics to plot (exclude AUC if it has None values)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results_df[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8)
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing evaluation module...")
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100, 2)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
    print_metrics(metrics, "Test Model")
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, "Test Confusion Matrix")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Get classification report
    report = get_classification_report(y_true, y_pred)
    print(f"\nClassification Report:\n{pd.DataFrame(report).T}")
    
    print("\n✅ Evaluation module test complete!")
