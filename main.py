"""
Main script to run complete baseline research pipeline
"""
import os
import shutil

# Force CPU mode if GPU has memory issues
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Uncomment this line to force CPU mode
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
import pandas as pd
from datetime import datetime

from config import Config
from dataset import load_dataset, create_dataloaders
from train import train_model, CheckpointManager
from evaluate import evaluate_all_strategies, export_results_to_excel, create_performance_charts, save_confusion_matrices
from visualization import print_dataset_statistics


def get_next_run_folder(base_results_dir):
    """
    Tạo folder mới cho mỗi lần chạy
    Tự động tăng số thứ tự: results/1/, results/2/, results/3/, ...
    
    Args:
        base_results_dir: Thư mục results gốc
    
    Returns:
        run_folder: Đường dẫn đến folder cho lần chạy này
        run_number: Số thứ tự lần chạy
    """
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Tìm tất cả các folder có dạng số
    existing_runs = []
    for item in os.listdir(base_results_dir):
        item_path = os.path.join(base_results_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            existing_runs.append(int(item))
    
    # Tìm số tiếp theo
    if existing_runs:
        next_run = max(existing_runs) + 1
    else:
        next_run = 1
    
    # Tạo folder mới
    run_folder = os.path.join(base_results_dir, str(next_run))
    os.makedirs(run_folder, exist_ok=True)
    
    return run_folder, next_run


def save_model_results(model_name, results, output_dir):
    """
    Save individual model results to Excel and create chart
    
    Args:
        model_name: Name of the model
        results: Dictionary with strategy results
        output_dir: Directory to save results
    """
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dataframe for this model
    rows = []
    for strategy_name, result in results.items():
        row = {
            'Model': model_name,
            'Strategy': strategy_name,
            **result['metrics']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns (including Test Loss)
    column_order = ['Model', 'Strategy', 'Test Loss', 'Accuracy (%)', 'Precision (%)', 
                   'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    df = df[column_order]
    
    # Save to Excel
    excel_path = os.path.join(model_dir, f'{model_name}_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"  ✓ Results saved to: {excel_path}")
    
    return df


def delete_model_checkpoints(model_name, checkpoints_dir):
    """
    Delete all checkpoints for a specific model
    
    Args:
        model_name: Name of the model
        checkpoints_dir: Base checkpoints directory
    """
    model_checkpoint_dir = os.path.join(checkpoints_dir, model_name)
    
    if os.path.exists(model_checkpoint_dir):
        try:
            shutil.rmtree(model_checkpoint_dir)
            print(f"  ✓ Deleted checkpoints: {model_checkpoint_dir}")
        except Exception as e:
            print(f"  ✗ Error deleting checkpoints: {str(e)}")
    else:
        print(f"  ⚠ No checkpoints found at: {model_checkpoint_dir}")


def main():
    """
    Main pipeline (Process each model sequentially to save disk space):
    1. Validate configuration
    2. Load and split dataset
    3. FOR EACH MODEL:
       - Train model
       - Evaluate with 3 strategies
       - Save individual results
       - Delete checkpoints
    4. Combine all results to Excel
    5. Generate combined performance charts
    """
    
    print("\n" + "="*70)
    print(" BASELINE RESEARCH - PRETRAINED MODELS EVALUATION")
    print("="*70)
    
    # Set random seeds for reproducibility across ALL models
    print(f"\n🔒 Setting random seeds for reproducibility (seed={Config.RANDOM_SEED})...")
    import random
    import numpy as np
    random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
        # For CUDA reproducibility (may impact performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=False)
    print("✓ Random seeds set successfully")
    
    # Step 1: Validate configuration
    print("\n[Step 1/6] Validating configuration...")
    Config.validate_config()
    
    # Tạo folder riêng cho lần chạy này
    run_folder, run_number = get_next_run_folder(Config.RESULTS_DIR)
    print(f"\n📁 Lần chạy thứ: {run_number}")
    print(f"📁 Kết quả sẽ được lưu tại: {run_folder}")
    
    # Create output directories
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Running in CPU mode. Training will be slower but uses less memory.")
        print("  To enable GPU: Increase Windows virtual memory (paging file) to 16-32GB")
    
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
    
    # Step 3: Train and evaluate each model (one at a time to save disk space)
    print(f"\n[Step 3/6] Training and evaluating {len(Config.MODELS)} models...")
    print("  Strategy: Train → Evaluate → Save Results → Delete Checkpoints")
    
    # Display dataset statistics once before training
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    
    print_dataset_statistics(train_paths + val_paths + test_paths, 
                           train_labels + val_labels + test_labels, 
                           class_names)
    
    all_model_results = {}
    successfully_processed = []
    
    for idx, model_name in enumerate(Config.MODELS, 1):
        print(f"\n{'='*70}")
        print(f"[Model {idx}/{len(Config.MODELS)}] Processing: {model_name}")
        print(f"{'='*70}")
        
        try:
            # 3.1: Train model
            print(f"\n  [3.1] Training {model_name}...")
            checkpoint_manager, history = train_model(
                model_name,
                train_loader,
                val_loader,
                num_classes,
                device,
                class_names=class_names
            )
            print(f"  ✓ Training completed for {model_name}")
            
            # 3.2: Evaluate with 3 strategies
            print(f"\n  [3.2] Evaluating {model_name} with 3 strategies...")
            results = evaluate_all_strategies(
                model_name,
                checkpoint_manager,
                test_loader,
                train_loader,  # CRITICAL: Pass train_loader for BatchNorm update
                num_classes,
                device,
                class_names=class_names
            )
            all_model_results[model_name] = results
            print(f"  ✓ Evaluation completed for {model_name}")
            
            # 3.3: Save individual model results
            print(f"\n  [3.3] Saving results for {model_name}...")
            save_model_results(model_name, results, run_folder)

            # 3.3.1: Save confusion matrices for this model
            model_dir = os.path.join(run_folder, model_name)
            save_confusion_matrices(
                {model_name: results}, model_dir, class_names=class_names
            )
            
            # 3.4: Delete checkpoints to free disk space
            print(f"\n  [3.4] Cleaning up checkpoints for {model_name}...")
            delete_model_checkpoints(model_name, Config.CHECKPOINTS_DIR)
            
            successfully_processed.append(model_name)
            print(f"\n  ✅ {model_name} completed successfully!")
            
        except Exception as e:
            print(f"\n  ✗ Error processing {model_name}: {str(e)}")
            print(f"  Skipping {model_name} and continuing with next model...")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_model_results:
        print("\n✗ No models processed successfully. Exiting...")
        return
    
    print(f"\n{'='*70}")
    print(f"✓ Successfully processed {len(successfully_processed)}/{len(Config.MODELS)} models")
    print(f"  Models: {', '.join(successfully_processed)}")
    print(f"{'='*70}")
    
    # Step 4: Combine all results to single Excel
    print(f"\n[Step 4/6] Combining all results to Excel...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(run_folder, f'all_models_results.xlsx')
    
    df = export_results_to_excel(all_model_results, excel_path)
    
    # Display summary
    print("\n" + "="*70)
    print("COMBINED RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    # Step 5: Generate combined performance charts
    print(f"\n[Step 5/6] Generating combined performance chart...")
    create_performance_charts(df, run_folder)
    
    # Step 6: Save experiment info
    print(f"\n[Step 6/6] Saving experiment metadata...")
    experiment_info = {
        'run_number': run_number,
        'timestamp': timestamp,
        'dataset_path': Config.DATASET_PATH,
        'num_classes': num_classes,
        'class_names': class_names,
        'train_samples': len(train_paths),
        'val_samples': len(val_paths),
        'test_samples': len(test_paths),
        'models_trained': successfully_processed,
        'config': {
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'early_stopping_patience': Config.EARLY_STOPPING_PATIENCE,
            'classifier_config': Config.CLASSIFIER_CONFIG,
            'dropout_rate': Config.DROPOUT_RATE
        }
    }
    
    info_path = os.path.join(run_folder, f'experiment_info.json')
    with open(info_path, 'w') as f:
        json.dump(experiment_info, f, indent=4)
    
    print(f"\n✓ Experiment info saved to: {info_path}")
    
    # Final summary
    print("\n" + "="*70)
    print(" BASELINE RESEARCH COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Lần chạy #{run_number}:")
    print(f"  - Folder: {run_folder}")
    print(f"  - Combined Excel: {excel_path}")
    print(f"  - Combined Chart: {os.path.join(run_folder, 'performance_comparison.png')}")
    print(f"  - Experiment Info: {info_path}")
    print(f"  - Individual Results: {run_folder}/<model_name>/")
    print(f"\n💾 Disk Space Optimization:")
    print(f"  - All checkpoints deleted after evaluation")
    print(f"  - Only results (Excel + Charts) kept")
    print(f"  - Estimated space saved: ~160GB (checkpoints)")
    
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
