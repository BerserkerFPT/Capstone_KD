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


def update_bn(model, train_loader, device, num_batches=100):
    """
    Update BatchNorm running statistics after loading averaged weights
    
    IMPORTANT: For frozen backbone models, we should NOT update the backbone BN layers
    because they already have good statistics from ImageNet pretraining.
    We only need to update BN layers in the classifier (if any).
    
    However, since our custom classifiers don't use BatchNorm, 
    we can skip this step entirely for most cases.
    
    For safety, we only update BN layers that are in trainable (unfrozen) parts.
    
    Args:
        model: Model with averaged weights
        train_loader: Training data loader
        device: Device to run on
        num_batches: Number of batches to use for BN update (default 100)
    """
    # First, identify which BN layers are in trainable parts
    trainable_bn_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            # Check if this BN layer has trainable parameters
            has_trainable = False
            for param in module.parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            
            if has_trainable:
                trainable_bn_layers.append((name, module))
    
    # If no trainable BN layers, skip update entirely
    if not trainable_bn_layers:
        print(f"      (No trainable BN layers found, skipping BN update)")
        return
    
    print(f"      (Found {len(trainable_bn_layers)} trainable BN layers to update)")
    
    # Set model to eval mode first
    model.eval()
    
    # Only set trainable BN layers to train mode and reset their statistics
    for name, module in trainable_bn_layers:
        module.train()
        module.momentum = None  # Use cumulative moving average
        module.reset_running_stats()
    
    # Forward pass to accumulate BN statistics (no gradient computation)
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    
    # Set everything back to eval mode
    model.eval()


def average_weights(checkpoint_paths, device):
    """
    Average model weights from multiple checkpoints
    
    IMPORTANT FOR FROZEN BACKBONE MODELS:
    - Frozen backbone weights are IDENTICAL across all checkpoints (they don't change during training)
    - Only the classifier/head weights differ between checkpoints
    - BatchNorm running statistics (running_mean, running_var) should NOT be averaged
      because they track population statistics, not learned parameters
    
    This function averages ALL learnable weights (including frozen ones, which are identical anyway)
    and keeps the BatchNorm running statistics from the FIRST checkpoint.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        device: Device to load checkpoints on
    
    Returns:
        averaged_state_dict: Averaged state dictionary
    """
    
    if not checkpoint_paths:
        return None
    
    if len(checkpoint_paths) == 1:
        # Only one checkpoint, no need to average
        checkpoint = torch.load(checkpoint_paths[0], map_location=device)
        return checkpoint['model_state_dict']
    
    # Load first checkpoint as base
    first_checkpoint = torch.load(checkpoint_paths[0], map_location=device)
    averaged_state_dict = copy.deepcopy(first_checkpoint['model_state_dict'])
    
    # Identify keys to average vs keys to keep from first checkpoint
    keys_to_average = []
    keys_to_keep = []
    
    for key in averaged_state_dict.keys():
        # Skip BatchNorm running statistics - these should NOT be averaged
        # They are population statistics, not learned parameters
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            keys_to_keep.append(key)
        else:
            keys_to_average.append(key)
    
    # Sum weights from remaining checkpoints (only for keys_to_average)
    for checkpoint_path in checkpoint_paths[1:]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        for key in keys_to_average:
            averaged_state_dict[key] = averaged_state_dict[key] + state_dict[key]
    
    # Compute average
    num_checkpoints = len(checkpoint_paths)
    for key in keys_to_average:
        averaged_state_dict[key] = averaged_state_dict[key] / num_checkpoints
    
    # keys_to_keep already have values from first checkpoint (no changes needed)
    
    return averaged_state_dict


def evaluate_model(model, test_loader, device, num_classes):
    """
    Evaluate model and compute metrics including test loss
    
    Returns:
        metrics: Dictionary with test_loss, accuracy, precision, recall, f1, auc
    """
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    total = 0
    
    # Create criterion for test loss calculation
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Accumulate loss
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute test loss
    test_loss = running_loss / total
    
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
        'Test Loss': test_loss,
        'Accuracy (%)': accuracy,
        'Precision (%)': precision,
        'Recall (%)': recall,
        'F1-Score (%)': f1,
        'AUC (%)': auc
    }
    
    return metrics


def strategy_1_best_checkpoint(model_name, checkpoint_manager, test_loader, num_classes, device, save_dir=None):
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
    
    # Save checkpoint if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'strategy1_best_epoch_{epoch}.pth')
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, save_path)
        print(f"    ✓ Saved Strategy 1 checkpoint to: {save_path}")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, num_classes)
    
    # Hiển thị chi tiết kết quả
    print(f"    {'='*60}")
    print(f"    📊 TEST RESULTS - Strategy 1:")
    print(f"    {'='*60}")
    print(f"    Test Loss : {metrics['Test Loss']:>6.4f}")
    print(f"    Accuracy  : {metrics['Accuracy (%)']:>6.2f}%")
    print(f"    Precision : {metrics['Precision (%)']:>6.2f}%")
    print(f"    Recall    : {metrics['Recall (%)']:>6.2f}%")
    print(f"    F1-Score  : {metrics['F1-Score (%)']:>6.2f}%")
    print(f"    AUC       : {metrics['AUC (%)']:>6.2f}%")
    print(f"    {'='*60}")
    
    return metrics


def strategy_2_top_k_average(model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, save_dir=None):
    """
    Strategy 2: Average top-K checkpoints and evaluate
    CRITICAL: Update BatchNorm stats after loading averaged weights
    
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
        model.load_state_dict(averaged_weights, strict=True)  # Use strict=True since we handle all keys properly
        model = model.to(device)
        
        # CRITICAL: Update BatchNorm statistics with training data
        print(f"      Updating BatchNorm statistics...")
        update_bn(model, train_loader, device, num_batches=100)
        
        # Save checkpoint if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'strategy2_top_{k}_averaged.pth')
            torch.save({'model_state_dict': model.state_dict(), 'k': k}, save_path)
            print(f"      ✓ Saved Strategy 2 (K={k}) checkpoint to: {save_path}")

        # Evaluate
        metrics = evaluate_model(model, test_loader, device, num_classes)
        results[k] = metrics
        
        # Hiển thị chi tiết kết quả
        print(f"      {'-'*56}")
        print(f"      📊 TEST RESULTS - Strategy 2 (K={k}):")
        print(f"      {'-'*56}")
        print(f"      Test Loss : {metrics['Test Loss']:>6.4f}")
        print(f"      Accuracy  : {metrics['Accuracy (%)']:>6.2f}%")
        print(f"      Precision : {metrics['Precision (%)']:>6.2f}%")
        print(f"      Recall    : {metrics['Recall (%)']:>6.2f}%")
        print(f"      F1-Score  : {metrics['F1-Score (%)']:>6.2f}%")
        print(f"      AUC       : {metrics['AUC (%)']:>6.2f}%")
        print(f"      {'-'*56}")
    
    return results


def strategy_3_last_n_average(model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, save_dir=None):
    """
    Strategy 3: Average last N epoch checkpoints
    CRITICAL: This is the most important strategy - must update BatchNorm stats!
    
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
    model.load_state_dict(averaged_weights, strict=True)  # Use strict=True since we handle all keys properly
    model = model.to(device)
    
    # CRITICAL: Update BatchNorm statistics with training data
    # This is ESSENTIAL because frozen backbone may have different BN stats across epochs
    print(f"    Updating BatchNorm statistics (this ensures model correctness)...")
    update_bn(model, train_loader, device, num_batches=100)
    
    # Save checkpoint if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'strategy3_last_{Config.LAST_N_EPOCHS}_averaged.pth')
        torch.save({'model_state_dict': model.state_dict(), 'epochs': epochs}, save_path)
        print(f"    ✓ Saved Strategy 3 checkpoint to: {save_path}")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, num_classes)
    
    # Hiển thị chi tiết kết quả
    print(f"    {'='*60}")
    print(f"    📊 TEST RESULTS - Strategy 3:")
    print(f"    {'='*60}")
    print(f"    Test Loss : {metrics['Test Loss']:>6.4f}")
    print(f"    Accuracy  : {metrics['Accuracy (%)']:>6.2f}%")
    print(f"    Precision : {metrics['Precision (%)']:>6.2f}%")
    print(f"    Recall    : {metrics['Recall (%)']:>6.2f}%")
    print(f"    F1-Score  : {metrics['F1-Score (%)']:>6.2f}%")
    print(f"    AUC       : {metrics['AUC (%)']:>6.2f}%")
    print(f"    {'='*60}")
    
    return metrics


def evaluate_all_strategies(model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, save_dir=None):
    """
    Evaluate all 3 strategies for a model
    
    Args:
        train_loader: Training data loader (needed for BatchNorm update in averaging strategies)
        save_dir: Directory to save averaged checkpoints
    
    Returns:
        all_results: Dictionary with all strategy results
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    
    all_results = {}
    
    # Prepare save directory for checkpoints
    cp_save_dir = None
    if save_dir:
        cp_save_dir = os.path.join(save_dir, 'saved_checkpoints')
        os.makedirs(cp_save_dir, exist_ok=True)

    # Strategy 1: Best single checkpoint (no averaging, no BN update needed)
    all_results['Strategy 1'] = strategy_1_best_checkpoint(
        model_name, checkpoint_manager, test_loader, num_classes, device, save_dir=cp_save_dir
    )
    
    # Strategy 2: Top-K averaging (with BN update)
    strategy_2_results = strategy_2_top_k_average(
        model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, save_dir=cp_save_dir
    )
    for k, metrics in strategy_2_results.items():
        all_results[f'Strategy 2 (K={k})'] = metrics
    
    # Strategy 3: Last N epochs averaging (with BN update - MOST CRITICAL)
    all_results['Strategy 3'] = strategy_3_last_n_average(
        model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, save_dir=cp_save_dir
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
    column_order = ['Model', 'Strategy', 'Test Loss', 'Accuracy (%)', 'Precision (%)', 
                   'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    df = df[column_order]
    
    # Save to Excel
    csv_path = output_path.replace('.xlsx', '.csv')
    df.to_csv(csv_path, index=False, sep=',', decimal=',')
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
    fig.suptitle('Model Performance Comparison Across All Strategies', 
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
