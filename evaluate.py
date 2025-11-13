"""
Evaluation script with 3 strategies and result export
"""
import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)

from config import Config
from models import get_model


def average_weights(checkpoint_paths, device):
    """
    Average model weights from multiple checkpoints
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        device: Device to load checkpoints on
    
    Returns:
        averaged_state_dict: Averaged state dictionary
    """
    
    if not checkpoint_paths:
        return None
    
    # Load first checkpoint
    first_checkpoint = torch.load(checkpoint_paths[0], map_location=device)
    averaged_state_dict = copy.deepcopy(first_checkpoint['model_state_dict'])
    
    # Add remaining checkpoints
    for checkpoint_path in checkpoint_paths[1:]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += state_dict[key]
    
    # Average
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = averaged_state_dict[key] / len(checkpoint_paths)
    
    return averaged_state_dict


def evaluate_model(model, test_loader, device, num_classes):
    """
    Evaluate model and compute metrics
    
    Returns:
        metrics: Dictionary with accuracy, precision, recall, f1, auc
    """
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except:
        auc = 0.0
    
    metrics = {
        'Accuracy (%)': accuracy,
        'Precision (%)': precision,
        'Recall (%)': recall,
        'F1-Score (%)': f1,
        'AUC (%)': auc
    }
    
    return metrics


def strategy_1_best_checkpoint(model_name, checkpoint_manager, test_loader, num_classes, device):
    """
    Strategy 1: Evaluate best checkpoint based on lowest val_loss
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    
    print(f"\n  Strategy 1: Best checkpoint (lowest val_loss)")
    
    # Get best checkpoint
    epoch, val_loss, checkpoint_path = checkpoint_manager.get_best_checkpoint()
    print(f"    Best checkpoint: Epoch {epoch}, Val Loss: {val_loss:.4f}")
    
    # Load model
    model = get_model(model_name, num_classes, freeze_backbone=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, num_classes)
    
    print(f"    Results: Acc={metrics['Accuracy (%)']:.2f}%, F1={metrics['F1-Score (%)']:.2f}%")
    
    return metrics


def strategy_2_top_k_average(model_name, checkpoint_manager, test_loader, num_classes, device):
    """
    Strategy 2: Average top-K checkpoints and evaluate
    
    Returns:
        results: Dictionary with k as key and metrics as value
    """
    
    print(f"\n  Strategy 2: Top-K checkpoint averaging")
    
    results = {}
    
    for k in Config.TOP_K_VALUES:
        print(f"    K={k}:")
        
        # Get top K checkpoints
        top_k = checkpoint_manager.get_top_k_checkpoints(k)
        
        if len(top_k) < k:
            print(f"      Warning: Only {len(top_k)} checkpoints available")
        
        checkpoint_paths = [path for _, _, path in top_k]
        
        # Average weights
        averaged_weights = average_weights(checkpoint_paths, device)
        
        # Load model with averaged weights
        model = get_model(model_name, num_classes, freeze_backbone=True)
        model.load_state_dict(averaged_weights)
        model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device, num_classes)
        results[k] = metrics
        
        print(f"      Results: Acc={metrics['Accuracy (%)']:.2f}%, F1={metrics['F1-Score (%)']:.2f}%")
    
    return results


def strategy_3_last_n_average(model_name, checkpoint_manager, test_loader, num_classes, device):
    """
    Strategy 3: Average last N epoch checkpoints
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    
    print(f"\n  Strategy 3: Last {Config.LAST_N_EPOCHS} epochs averaging")
    
    # Get last N checkpoints
    last_n = checkpoint_manager.get_last_n_checkpoints(Config.LAST_N_EPOCHS)
    
    if len(last_n) < Config.LAST_N_EPOCHS:
        print(f"    Warning: Only {len(last_n)} checkpoints available")
    
    checkpoint_paths = [path for _, _, path in last_n]
    epochs = [epoch for epoch, _, _ in last_n]
    print(f"    Averaging epochs: {epochs}")
    
    # Average weights
    averaged_weights = average_weights(checkpoint_paths, device)
    
    # Load model with averaged weights
    model = get_model(model_name, num_classes, freeze_backbone=True)
    model.load_state_dict(averaged_weights)
    model = model.to(device)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, num_classes)
    
    print(f"    Results: Acc={metrics['Accuracy (%)']:.2f}%, F1={metrics['F1-Score (%)']:.2f}%")
    
    return metrics


def evaluate_all_strategies(model_name, checkpoint_manager, test_loader, num_classes, device):
    """
    Evaluate all 3 strategies for a model
    
    Returns:
        all_results: Dictionary with all strategy results
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    
    all_results = {}
    
    # Strategy 1
    all_results['Strategy 1'] = strategy_1_best_checkpoint(
        model_name, checkpoint_manager, test_loader, num_classes, device
    )
    
    # Strategy 2
    strategy_2_results = strategy_2_top_k_average(
        model_name, checkpoint_manager, test_loader, num_classes, device
    )
    for k, metrics in strategy_2_results.items():
        all_results[f'Strategy 2 (K={k})'] = metrics
    
    # Strategy 3
    all_results['Strategy 3'] = strategy_3_last_n_average(
        model_name, checkpoint_manager, test_loader, num_classes, device
    )
    
    return all_results


def export_results_to_excel(all_model_results, output_path):
    """
    Export all results to Excel file
    
    Args:
        all_model_results: Dictionary with model_name as key and strategy results as value
        output_path: Path to save Excel file
    """
    
    rows = []
    
    for model_name, strategy_results in all_model_results.items():
        for strategy_name, metrics in strategy_results.items():
            row = {
                'Model': model_name,
                'Strategy': strategy_name,
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = ['Model', 'Strategy', 'Accuracy (%)', 'Precision (%)', 
                   'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    df = df[column_order]
    
    # Save to Excel
    df.to_excel(output_path, index=False)
    print(f"\n✓ Results exported to: {output_path}")
    
    return df


def create_performance_charts(df, output_dir):
    """
    Create single comprehensive performance comparison chart
    
    Args:
        df: Results dataframe
        output_dir: Directory to save chart
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive comparison chart
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Performance Comparison Across 3 Strategies', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Prepare data for grouped bar chart
        models = df['Model'].unique()
        strategies = df['Strategy'].unique()
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, strategy in enumerate(strategies):
            strategy_data = df[df['Strategy'] == strategy]
            values = [strategy_data[strategy_data['Model'] == model][metric].values[0] 
                     for model in models]
            ax.bar(x + i * width, values, width, label=strategy)
        
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    # Summary table in the last subplot
    ax = axes[1, 2]
    ax.axis('off')
    
    # Find best performing model for each strategy
    summary_text = "Best Models per Strategy:\n\n"
    for strategy in df['Strategy'].unique():
        strategy_df = df[df['Strategy'] == strategy]
        best_idx = strategy_df['F1-Score (%)'].idxmax()
        best_model = strategy_df.loc[best_idx, 'Model']
        best_f1 = strategy_df.loc[best_idx, 'F1-Score (%)']
        summary_text += f"{strategy}:\n  {best_model} (F1: {best_f1:.2f}%)\n\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Chart saved to: {chart_path}")


if __name__ == "__main__":
    print("Evaluation script ready. Run main.py to execute full pipeline.")
