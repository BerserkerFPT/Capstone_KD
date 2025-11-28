"""
Test script for visualization features
Tests all visualization functions without running actual model training
"""
import os
import numpy as np
from visualization import display_sample_images, print_dataset_statistics, plot_training_history, plot_patience_period


def load_test_dataset(dataset_path):
    """
    Load images from dataset for testing
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        image_paths, labels, class_names
    """
    print(f"Loading test dataset from: {dataset_path}")
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
    
    # Collect all images
    image_paths = []
    labels = []
    
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
    
    print(f"Loaded {len(image_paths)} images total")
    
    return image_paths, labels, class_dirs


def generate_fake_training_history(num_epochs=50, early_stop_epoch=35):
    """
    Generate realistic fake training history for testing plots
    
    Args:
        num_epochs: Total number of epochs to simulate
        early_stop_epoch: Epoch where training stopped (for early stopping simulation)
        
    Returns:
        history: Dictionary with train/val loss and accuracy
    """
    print(f"\nGenerating simulated training history for {early_stop_epoch} epochs...")
    
    # Simulate training with realistic trends
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    np.random.seed(42)  # For reproducibility
    
    for epoch in range(1, early_stop_epoch + 1):
        # Training loss: decreases rapidly then plateaus
        train_loss = 1.5 * np.exp(-epoch / 10) + 0.1 + np.random.normal(0, 0.02)
        
        # Validation loss: similar but slightly higher and more noisy
        val_loss = 1.5 * np.exp(-epoch / 12) + 0.15 + np.random.normal(0, 0.03)
        
        # Training accuracy: increases then plateaus around 95%
        train_acc = 50 + 45 * (1 - np.exp(-epoch / 8)) + np.random.normal(0, 0.5)
        train_acc = min(98, max(50, train_acc))
        
        # Validation accuracy: similar but slightly lower
        val_acc = 50 + 42 * (1 - np.exp(-epoch / 10)) + np.random.normal(0, 1.0)
        val_acc = min(95, max(50, val_acc))
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    print(f"✓ Generated history for {len(history['train_loss'])} epochs")
    return history


def test_visualization_functions():
    """
    Test all visualization functions
    """
    print("\n" + "="*70)
    print(" TESTING VISUALIZATION FEATURES")
    print("="*70)
    
    # Configuration
    DATASET_PATH = r"D:\Rice_photos"
    OUTPUT_DIR = "test_visualizations"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Test 1: Load dataset
    print("\n" + "="*70)
    print("TEST 1: Loading Dataset")
    print("="*70)
    
    try:
        image_paths, labels, class_names = load_test_dataset(DATASET_PATH)
        print(f"✓ Dataset loaded successfully")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Classes: {class_names}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and contains subdirectories for each class")
        return
    
    # Test 2: Print dataset statistics
    print("\n" + "="*70)
    print("TEST 2: Dataset Statistics")
    print("="*70)
    
    try:
        print_dataset_statistics(image_paths, labels, class_names)
        print("✓ Dataset statistics printed successfully")
    except Exception as e:
        print(f"✗ Error printing statistics: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Display sample images
    print("\n" + "="*70)
    print("TEST 3: Sample Images Display")
    print("="*70)
    
    try:
        sample_path = os.path.join(OUTPUT_DIR, "test_sample_images.png")
        display_sample_images(image_paths, labels, class_names, 
                            samples_per_class=4, 
                            save_path=sample_path)
        print("✓ Sample images displayed and saved successfully")
    except Exception as e:
        print(f"✗ Error displaying sample images: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Generate and plot training history
    print("\n" + "="*70)
    print("TEST 4: Training History Plot")
    print("="*70)
    
    try:
        # Generate fake training data
        history = generate_fake_training_history(num_epochs=50, early_stop_epoch=35)
        
        # Plot full training history
        history_path = os.path.join(OUTPUT_DIR, "test_training_history.png")
        plot_training_history(history, "test_model", save_path=history_path)
        print("✓ Training history plot generated successfully")
    except Exception as e:
        print(f"✗ Error plotting training history: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Plot patience period
    print("\n" + "="*70)
    print("TEST 5: Patience Period Plot")
    print("="*70)
    
    try:
        # Plot patience period (last 15 epochs)
        patience_path = os.path.join(OUTPUT_DIR, "test_patience_period.png")
        plot_patience_period(history, patience=15, model_name="test_model", 
                           save_path=patience_path)
        print("✓ Patience period plot generated successfully")
    except Exception as e:
        print(f"✗ Error plotting patience period: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print(f"\n✓ All visualization tests completed!")
    print(f"\nGenerated files in '{OUTPUT_DIR}':")
    for file in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  - {file} ({size_kb:.1f} KB)")
    
    print(f"\n📁 You can view the test results in: {os.path.abspath(OUTPUT_DIR)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_visualization_functions()
