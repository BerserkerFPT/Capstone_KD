"""
Visualization utilities for training analysis
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def print_dataset_statistics(image_paths, labels, class_names):

    """
    Print comprehensive dataset statistics including image counts and sizes
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        class_names: List of class names
    """
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    
    # Total images
    total_images = len(image_paths)
    print(f"\nTotal Images: {total_images}")
    
    # Images per class
    print(f"\nImages per Class:")
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    # Analyze image sizes
    print(f"\nAnalyzing image sizes...")
    widths = []
    heights = []
    
    # Sample images to avoid loading all (can be slow for large datasets)
    sample_size = min(1000, len(image_paths))
    sampled_paths = random.sample(image_paths, sample_size)
    
    for img_path in sampled_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            print(f"  Warning: Could not read {img_path}: {e}")
    
    if widths and heights:
        print(f"\nImage Size Statistics (sampled {len(widths)} images):")
        print(f"  Width  - Min: {min(widths)}px, Max: {max(widths)}px, Mean: {np.mean(widths):.1f}px")
        print(f"  Height - Min: {min(heights)}px, Max: {max(heights)}px, Mean: {np.mean(heights):.1f}px")
        
        # Check if images have uniform sizes
        if len(set(zip(widths, heights))) == 1:
            print(f"  ✓ All images have uniform size: {widths[0]}x{heights[0]}")
        else:
            unique_sizes = len(set(zip(widths, heights)))
            print(f"  ⚠ Images have {unique_sizes} different sizes")
    
    print("="*70)


def plot_training_history(history, model_name, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='training', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'orange', label='validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='training', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'orange', label='validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to: {save_path}")
    
    plt.close()
    
    return fig


def plot_patience_period(history, patience, model_name, save_path=None):
    """
    Plot training metrics during the early stopping patience period
    
    Args:
        history: Dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        patience: Early stopping patience value
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    total_epochs = len(history['train_loss'])
    
    # Extract last N epochs based on patience (or all epochs if fewer than patience)
    start_epoch = max(0, total_epochs - patience)
    epochs = range(start_epoch + 1, total_epochs + 1)
    
    train_loss = history['train_loss'][start_epoch:]
    val_loss = history['val_loss'][start_epoch:]
    train_acc = history['train_acc'][start_epoch:]
    val_acc = history['val_acc'][start_epoch:]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Loss
    ax1.plot(epochs, train_loss, 'b-', label='training', linewidth=2, marker='o')
    ax1.plot(epochs, val_loss, 'orange', label='validation', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Model Loss (Last {len(epochs)} Epochs - Patience Period)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='training', linewidth=2, marker='o')
    ax2.plot(epochs, val_acc, 'orange', label='validation', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Model Accuracy (Last {len(epochs)} Epochs - Patience Period)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Patience period plot saved to: {save_path}")
    
    plt.close()
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module loaded successfully!")
