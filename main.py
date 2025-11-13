"""
Main script to run complete baseline research pipeline
"""
import os
import torch
import json
from datetime import datetime

from config import Config
from dataset import load_dataset, create_dataloaders
from train import train_model, CheckpointManager
from evaluate import evaluate_all_strategies, export_results_to_excel, create_performance_charts


def main():
    """
    Main pipeline:
    1. Validate configuration
    2. Load and split dataset
    3. Train all models
    4. Evaluate all models with 3 strategies
    5. Export results to Excel
    6. Generate performance charts
    """
    
    print("\n" + "="*70)
    print(" BASELINE RESEARCH - PRETRAINED MODELS EVALUATION")
    print("="*70)
    
    # Step 1: Validate configuration
    print("\n[Step 1/6] Validating configuration...")
    Config.validate_config()
    
    # Create output directories
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 2: Load dataset
    print("\n[Step 2/6] Loading and splitting dataset...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = load_dataset(
        Config.DATASET_PATH,
        Config.TRAIN_RATIO,
        Config.VAL_RATIO,
        Config.TEST_RATIO,
        Config.RANDOM_SEED
    )
    
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )
    
    # Step 3: Train all models
    print(f"\n[Step 3/6] Training {len(Config.MODELS)} models...")
    checkpoint_managers = {}
    
    for model_name in Config.MODELS:
        try:
            checkpoint_manager = train_model(
                model_name,
                train_loader,
                val_loader,
                num_classes,
                device
            )
            checkpoint_managers[model_name] = checkpoint_manager
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            print(f"  Skipping {model_name}...")
            continue
    
    if not checkpoint_managers:
        print("\n✗ No models trained successfully. Exiting...")
        return
    
    print(f"\n✓ Successfully trained {len(checkpoint_managers)}/{len(Config.MODELS)} models")
    
    # Step 4: Evaluate all models with all strategies
    print(f"\n[Step 4/6] Evaluating models with 3 strategies...")
    all_model_results = {}
    
    for model_name, checkpoint_manager in checkpoint_managers.items():
        try:
            results = evaluate_all_strategies(
                model_name,
                checkpoint_manager,
                test_loader,
                num_classes,
                device
            )
            all_model_results[model_name] = results
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {str(e)}")
            continue
    
    # Step 5: Export results to Excel
    print(f"\n[Step 5/6] Exporting results to Excel...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(Config.RESULTS_DIR, f'baseline_results_{timestamp}.xlsx')
    
    df = export_results_to_excel(all_model_results, excel_path)
    
    # Display summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    # Step 6: Generate performance charts
    print(f"\n[Step 6/6] Generating performance chart...")
    create_performance_charts(df, Config.RESULTS_DIR)
    
    # Save experiment info
    experiment_info = {
        'timestamp': timestamp,
        'dataset_path': Config.DATASET_PATH,
        'num_classes': num_classes,
        'class_names': class_names,
        'train_samples': len(train_paths),
        'val_samples': len(val_paths),
        'test_samples': len(test_paths),
        'models_trained': list(checkpoint_managers.keys()),
        'config': {
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'early_stopping_patience': Config.EARLY_STOPPING_PATIENCE,
            'classifier_config': Config.CLASSIFIER_CONFIG,
            'dropout_rate': Config.DROPOUT_RATE
        }
    }
    
    info_path = os.path.join(Config.RESULTS_DIR, f'experiment_info_{timestamp}.json')
    with open(info_path, 'w') as f:
        json.dump(experiment_info, f, indent=4)
    
    print(f"\n✓ Experiment info saved to: {info_path}")
    
    # Final summary
    print("\n" + "="*70)
    print(" BASELINE RESEARCH COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"  - Excel file: {excel_path}")
    print(f"  - Performance chart: {os.path.join(Config.RESULTS_DIR, 'performance_comparison.png')}")
    print(f"  - Experiment info: {info_path}")
    print(f"  - Checkpoints: {Config.CHECKPOINTS_DIR}")
    
    # Find best model (based on Strategy 1 F1-Score)
    strategy_1_df = df[df['Strategy'] == 'Strategy 1']
    best_idx = strategy_1_df['F1-Score (%)'].idxmax()
    best_model = strategy_1_df.loc[best_idx, 'Model']
    best_f1 = strategy_1_df.loc[best_idx, 'F1-Score (%)']
    best_acc = strategy_1_df.loc[best_idx, 'Accuracy (%)']
    
    print(f"\n🏆 Best Model (Strategy 1):")
    print(f"  Model: {best_model}")
    print(f"  Accuracy: {best_acc:.2f}%")
    print(f"  F1-Score: {best_f1:.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
