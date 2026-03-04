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


def evaluate_model(model, test_loader, device, num_classes, class_names=None):
    """
    Evaluate model and compute metrics including test loss, per-class metrics, and confusion matrix

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        num_classes: Number of classes
        class_names: List of class names (optional, defaults to Class 0, Class 1, ...)

    Returns:
        result: Dictionary with keys:
            - 'metrics': Overall macro-averaged metrics
            - 'per_class': Per-class metrics dict {class_name: {metric: value}}
            - 'confusion_matrix': Confusion matrix as numpy array
    """

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

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

    # Compute macro-averaged metrics
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

    # ========== Per-class metrics ==========
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Per-class precision, recall, f1 from sklearn
    pc_precision = precision_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0) * 100
    pc_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0) * 100
    pc_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0) * 100

    per_class = {}
    for i in range(num_classes):
        cls_name = class_names[i]

        # Derive TP, FP, FN, TN from confusion matrix
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        # Specificity = TN / (TN + FP)
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0

        # Support = number of true samples for this class
        support = int(cm[i, :].sum())

        # Per-class AUC (one-vs-rest)
        try:
            binary_labels = (all_labels == i).astype(int)
            class_auc = roc_auc_score(binary_labels, all_probs[:, i]) * 100
        except:
            class_auc = 0.0

        per_class[cls_name] = {
            'Precision (%)': pc_precision[i],
            'Recall (%)': pc_recall[i],
            'F1-Score (%)': pc_f1[i],
            'Specificity (%)': specificity,
            'AUC (%)': class_auc,
            'Support': support
        }

    result = {
        'metrics': metrics,
        'per_class': per_class,
        'confusion_matrix': cm
    }

    return result


def _print_eval_results(metrics, per_class, prefix="    ", header="TEST RESULTS"):
    """Helper to print macro and per-class evaluation results"""
    print(f"{prefix}{'='*60}")
    print(f"{prefix}📊 {header}:")
    print(f"{prefix}{'='*60}")
    print(f"{prefix}Test Loss : {metrics['Test Loss']:>6.4f}")
    print(f"{prefix}Accuracy  : {metrics['Accuracy (%)']:>6.2f}%")
    print(f"{prefix}Precision : {metrics['Precision (%)']:>6.2f}%")
    print(f"{prefix}Recall    : {metrics['Recall (%)']:>6.2f}%")
    print(f"{prefix}F1-Score  : {metrics['F1-Score (%)']:>6.2f}%")
    print(f"{prefix}AUC       : {metrics['AUC (%)']:>6.2f}%")

    # Per-class breakdown
    print(f"{prefix}{'-'*60}")
    print(f"{prefix}Per-Class Breakdown:")
    print(f"{prefix}{'-'*60}")
    header_fmt = f"{prefix}  {'Class':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Spec':>6} {'AUC':>6} {'Sup':>5}"
    print(header_fmt)
    print(f"{prefix}  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for cls_name, cls_metrics in per_class.items():
        print(f"{prefix}  {cls_name:<35} "
              f"{cls_metrics['Precision (%)']:>5.1f}% "
              f"{cls_metrics['Recall (%)']:>5.1f}% "
              f"{cls_metrics['F1-Score (%)']:>5.1f}% "
              f"{cls_metrics['Specificity (%)']:>5.1f}% "
              f"{cls_metrics['AUC (%)']:>5.1f}% "
              f"{cls_metrics['Support']:>5d}")
    print(f"{prefix}{'='*60}")


def strategy_1_best_checkpoint(model_name, checkpoint_manager, test_loader, num_classes, device, class_names=None):
    """
    Strategy 1: Evaluate best checkpoint based on lowest val_loss

    Returns:
        result: Dictionary with 'metrics', 'per_class', 'confusion_matrix'
    """

    print(f"\n  Strategy 1: Best checkpoint (lowest val_loss)")

    # Get best checkpoint
    epoch, val_loss, checkpoint_path = checkpoint_manager.get_best_checkpoint()
    print(f"    Best checkpoint: Epoch {epoch}, Val Loss: {val_loss:.4f}")

    # Load model
    model = get_model(model_name, num_classes, freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    result = evaluate_model(model, test_loader, device, num_classes, class_names)

    # Hiển thị chi tiết kết quả
    _print_eval_results(result['metrics'], result['per_class'], prefix="    ", header="TEST RESULTS - Strategy 1")

    return result


def strategy_2_top_k_average(model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, class_names=None):
    """
    Strategy 2: Average top-K checkpoints and evaluate
    CRITICAL: Update BatchNorm stats after loading averaged weights

    Returns:
        results: Dictionary with k as key and result dict as value
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
        model = get_model(model_name, num_classes, freeze_backbone=False)
        model.load_state_dict(averaged_weights, strict=True)  # Use strict=True since we handle all keys properly
        model = model.to(device)

        # CRITICAL: Update BatchNorm statistics with training data
        print(f"      Updating BatchNorm statistics...")
        update_bn(model, train_loader, device, num_batches=100)

        # Evaluate
        result = evaluate_model(model, test_loader, device, num_classes, class_names)
        results[k] = result

        # Hiển thị chi tiết kết quả
        _print_eval_results(result['metrics'], result['per_class'], prefix="      ", header=f"TEST RESULTS - Strategy 2 (K={k})")

    return results


def strategy_3_last_n_average(model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, class_names=None):
    """
    Strategy 3: Average last N epoch checkpoints
    CRITICAL: This is the most important strategy - must update BatchNorm stats!

    Returns:
        result: Dictionary with 'metrics', 'per_class', 'confusion_matrix'
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
    model = get_model(model_name, num_classes, freeze_backbone=False)
    model.load_state_dict(averaged_weights, strict=True)  # Use strict=True since we handle all keys properly
    model = model.to(device)
    # CRITICAL: Update BatchNorm statistics with training data
    # This is ESSENTIAL because frozen backbone may have different BN stats across epochs
    print(f"    Updating BatchNorm statistics (this ensures model correctness)...")
    update_bn(model, train_loader, device, num_batches=100)

    # Evaluate
    result = evaluate_model(model, test_loader, device, num_classes, class_names)

    # Hiển thị chi tiết kết quả
    _print_eval_results(result['metrics'], result['per_class'], prefix="    ", header="TEST RESULTS - Strategy 3")

    return result


def evaluate_all_strategies(model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, class_names=None):
    """
    Evaluate all 3 strategies for a model

    Args:
        model_name: Model name
        checkpoint_manager: Checkpoint manager
        test_loader: Test data loader
        train_loader: Training data loader (needed for BatchNorm update in averaging strategies)
        num_classes: Number of classes
        device: Device
        class_names: List of class names

    Returns:
        all_results: Dictionary with strategy name as key and result dict as value.
                     Each result dict has keys: 'metrics', 'per_class', 'confusion_matrix'
    """

    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")

    all_results = {}

    # Strategy 1: Best single checkpoint (no averaging, no BN update needed)
    all_results['Strategy 1'] = strategy_1_best_checkpoint(
        model_name, checkpoint_manager, test_loader, num_classes, device, class_names
    )

    # Strategy 2: Top-K averaging (with BN update)
    strategy_2_results = strategy_2_top_k_average(
        model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, class_names
    )
    for k, result in strategy_2_results.items():
        all_results[f'Strategy 2 (K={k})'] = result

    # Strategy 3: Last N epochs averaging (with BN update - MOST CRITICAL)
    all_results['Strategy 3'] = strategy_3_last_n_average(
        model_name, checkpoint_manager, test_loader, train_loader, num_classes, device, class_names
    )

    return all_results


def export_results_to_excel(all_model_results, output_path, class_names=None):
    """
    Export all results to Excel file with separate sheets for macro and per-class metrics

    Args:
        all_model_results: Dictionary with model_name as key and strategy results as value.
                           Each strategy result has keys: 'metrics', 'per_class', 'confusion_matrix'
        output_path: Path to save Excel file
        class_names: List of class names (for column ordering)

    Returns:
        df: Macro-averaged results dataframe
    """

    macro_rows = []
    per_class_rows = []

    for model_name, strategy_results in all_model_results.items():
        for strategy_name, result in strategy_results.items():
            # Macro metrics row
            macro_row = {
                'Model': model_name,
                'Strategy': strategy_name,
                **result['metrics']
            }
            macro_rows.append(macro_row)

            # Per-class metrics rows
            for cls_name, cls_metrics in result['per_class'].items():
                pc_row = {
                    'Model': model_name,
                    'Strategy': strategy_name,
                    'Class': cls_name,
                    **cls_metrics
                }
                per_class_rows.append(pc_row)

    # Macro dataframe
    df = pd.DataFrame(macro_rows)
    column_order = ['Model', 'Strategy', 'Test Loss', 'Accuracy (%)', 'Precision (%)',
                   'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    df = df[column_order]

    # Per-class dataframe
    df_pc = pd.DataFrame(per_class_rows)
    pc_column_order = ['Model', 'Strategy', 'Class', 'Precision (%)', 'Recall (%)',
                       'F1-Score (%)', 'Specificity (%)', 'AUC (%)', 'Support']
    df_pc = df_pc[pc_column_order]

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Overall Metrics', index=False)
        df_pc.to_excel(writer, sheet_name='Per-Class Metrics', index=False)

    print(f"\n✓ Results exported to: {output_path}")
    print(f"  - Sheet 'Overall Metrics': Macro-averaged metrics")
    print(f"  - Sheet 'Per-Class Metrics': Per-class breakdown")

    return df


def save_confusion_matrices(all_model_results, output_dir, class_names=None):
    """
    Save confusion matrix heatmaps for all models and strategies

    Args:
        all_model_results: Dictionary with model_name as key and strategy results as value
        output_dir: Directory to save confusion matrix images
        class_names: List of class names for axis labels
    """
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    for model_name, strategy_results in all_model_results.items():
        for strategy_name, result in strategy_results.items():
            cm = result['confusion_matrix']

            fig, ax = plt.subplots(figsize=(max(8, len(cm) * 1.2), max(6, len(cm) * 1.0)))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names if class_names else range(len(cm)),
                       yticklabels=class_names if class_names else range(len(cm)),
                       ax=ax)

            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)

            # Clean strategy name for filename
            safe_strategy = strategy_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            ax.set_title(f'{model_name} - {strategy_name}\nConfusion Matrix', fontsize=13, fontweight='bold')

            plt.tight_layout()

            filename = f'{model_name}_{safe_strategy}_cm.png'
            filepath = os.path.join(cm_dir, filename)
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()

    print(f"✓ Confusion matrices saved to: {cm_dir}")


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
