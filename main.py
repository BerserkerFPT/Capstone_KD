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
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

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
    Save individual model results to Excel (2 sheets: macro + per-class)

    Args:
        model_name: Name of the model
        results: Dictionary with strategy results
        output_dir: Directory to save results
    """
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Sheet 1: Macro-averaged metrics
    macro_rows = []
    for strategy_name, result in results.items():
        row = {
            'Model': model_name,
            'Strategy': strategy_name,
            **result['metrics']
        }
        macro_rows.append(row)

    df_macro = pd.DataFrame(macro_rows)

    # Reorder columns (including Test Loss)
    column_order = ['Model', 'Strategy', 'Test Loss', 'Accuracy (%)', 'Precision (%)',
                   'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    df_macro = df_macro[column_order]

    # Sheet 2: Per-class metrics
    per_class_rows = []
    for strategy_name, result in results.items():
        for cls_name, cls_metrics in result['per_class'].items():
            pc_row = {
                'Model': model_name,
                'Strategy': strategy_name,
                'Class': cls_name,
                **cls_metrics
            }
            per_class_rows.append(pc_row)

    df_per_class = pd.DataFrame(per_class_rows)
    pc_column_order = ['Model', 'Strategy', 'Class', 'Precision (%)', 'Recall (%)',
                       'F1-Score (%)', 'Specificity (%)', 'AUC (%)', 'Support']
    df_per_class = df_per_class[pc_column_order]

    # Save both sheets to Excel
    excel_path = os.path.join(model_dir, f'{model_name}_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_macro.to_excel(writer, sheet_name='Macro Results', index=False)
        df_per_class.to_excel(writer, sheet_name='Per-Class Results', index=False)
    print(f"  ✓ Results saved to: {excel_path}")

    return df_macro


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


def export_run_config(run_folder, num_classes=None, class_names=None, 
                      train_count=0, val_count=0, test_count=0):
    """
    Xuất toàn bộ config của lần chạy ra file Excel (run_config.xlsx).
    Mỗi nhóm config = 1 sheet riêng, dễ đọc và so sánh giữa các lần chạy.
    Nếu tắt focal loss → các param focal tự set 0/None.
    Nếu tắt WRS → ghi rõ DISABLED.
    
    Args:
        run_folder: Folder lưu kết quả của lần chạy
        num_classes: Số lượng class
        class_names: Danh sách tên class
        train_count, val_count, test_count: Số lượng ảnh mỗi split
    """
    is_focal = Config.LOSS_FUNCTION == 'poly_focal'
    
    # Sheet 1: Dataset & Splitting
    dataset_rows = [
        ("dataset_path", Config.DATASET_PATH),
        ("num_classes", num_classes),
        ("class_names", ", ".join(class_names) if class_names else ""),
        ("train_ratio", Config.TRAIN_RATIO),
        ("val_ratio", Config.VAL_RATIO),
        ("test_ratio", Config.TEST_RATIO),
        ("train_samples", train_count),
        ("val_samples", val_count),
        ("test_samples", test_count),
        ("image_size", Config.IMAGE_SIZE),
        ("random_seed", Config.RANDOM_SEED),
    ]
    
    # Sheet 2: Model
    model_rows = [
        ("models", ", ".join(Config.MODELS)),
        ("classifier_config", str(Config.CLASSIFIER_CONFIG)),
        ("dropout_rate", Config.DROPOUT_RATE),
    ]
    
    # Sheet 3: Training Hyperparameters
    training_rows = [
        ("batch_size", Config.BATCH_SIZE),
        ("num_epochs", Config.NUM_EPOCHS),
        ("learning_rate", Config.LEARNING_RATE),
        ("weight_decay", Config.WEIGHT_DECAY),
        ("warmup_epochs", Config.WARMUP_EPOCHS),
        ("eta_min (CosineAnnealing)", Config.ETA_MIN),
        ("num_workers", Config.NUM_WORKERS),
        ("early_stopping_patience", Config.EARLY_STOPPING_PATIENCE),
        ("lr_decay_patience", Config.LR_DECAY_PATIENCE),
        ("lr_decay_factor", Config.LR_DECAY_FACTOR),
    ]
    
    # Sheet 4: Loss Function
    loss_rows = [
        ("loss_function", Config.LOSS_FUNCTION),
        ("label_smoothing", Config.LABEL_SMOOTHING if not is_focal else 0),
        ("focal_gamma", Config.FOCAL_GAMMA if is_focal else 0),
        ("poly_epsilon", Config.POLY_EPSILON if is_focal else 0),
        ("class_weight_method", Config.CLASS_WEIGHT_METHOD if is_focal else "N/A"),
    ]
    
    # Sheet 5: Sampler & Cross-Validation
    sampler_cv_rows = [
        ("use_weighted_random_sampler", Config.USE_WEIGHTED_SAMPLER),
        ("use_cross_validation", Config.USE_CROSS_VALIDATION),
        ("cv_n_splits", Config.CV_N_SPLITS if Config.USE_CROSS_VALIDATION else 0),
    ]
    
    # Sheet 6: Evaluation & Output
    eval_rows = [
        ("top_k_values", str(Config.TOP_K_VALUES)),
        ("last_n_epochs", Config.LAST_N_EPOCHS),
        ("keep_last_n_checkpoints", Config.KEEP_LAST_N_CHECKPOINTS),
        ("keep_top_k_checkpoints", Config.KEEP_TOP_K_CHECKPOINTS),
        ("auto_delete_checkpoints", Config.AUTO_DELETE_CHECKPOINTS),
        ("experiment_name", Config.EXPERIMENT_NAME),
    ]
    
    # Sheet 7: Compatibility Notes
    notes_rows = [
        ("WRS + Focal Loss", 
         "BOTH ACTIVE - WRS handles imbalance at data level, Focal Loss at loss level. "
         "May double-correct for imbalance." 
         if (Config.USE_WEIGHTED_SAMPLER and is_focal) else "No conflict"),
        ("Metrics Averaging", 
         "All metrics (Precision, Recall, F1, AUC) use MACRO averaging. "
         "Accuracy is computed as overall (correct/total)."),
    ]
    
    config_path = os.path.join(run_folder, "run_config.xlsx")
    with pd.ExcelWriter(config_path, engine='openpyxl') as writer:
        for sheet_name, rows in [
            ("Dataset & Splitting", dataset_rows),
            ("Model", model_rows),
            ("Training Hyperparams", training_rows),
            ("Loss Function", loss_rows),
            ("Sampler & CV", sampler_cv_rows),
            ("Evaluation & Output", eval_rows),
            ("Notes", notes_rows),
        ]:
            df = pd.DataFrame(rows, columns=["Parameter", "Value"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"  ✓ Full run config exported to: {config_path}")
    return config_path


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
    # torch.use_deterministic_algorithms(True,warn_only=False)
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
    
    # Export full run config
    print("\n[Config] Exporting run configuration...")
    export_run_config(
        run_folder, 
        num_classes=num_classes, 
        class_names=class_names,
        train_count=len(train_paths),
        val_count=len(val_paths),
        test_count=len(test_paths)
    )
    
    # Warn if both WRS and Focal Loss are active
    if Config.USE_WEIGHTED_SAMPLER and Config.LOSS_FUNCTION == 'poly_focal':
        print("\n⚠ WARNING: Cả WeightedRandomSampler và PolyFocalLoss đều đang BẬT!")
        print("  → WRS xử lý imbalance ở data level (oversampling minority class)")
        print("  → Focal Loss xử lý imbalance ở loss level (focus on hard examples)")
        print("  → Có thể gây double-correction. Hãy cân nhắc tắt 1 trong 2 nếu kết quả không tốt.")
    
    # ===================== CROSS-VALIDATION MODE (Pure K-Fold) =====================
    if Config.USE_CROSS_VALIDATION:
        print(f"\n{'='*70}")
        print(f" PURE CROSS-VALIDATION ({Config.CV_N_SPLITS}-Fold Stratified)")
        print(f"{'='*70}")
        
        # Gộp toàn bộ data (train + val + test) → 1 pool duy nhất
        all_paths = train_paths + val_paths + test_paths
        all_labels = train_labels + val_labels + test_labels
        
        print(f"  Total data: {len(all_paths)} images → chia {Config.CV_N_SPLITS} fold")
        
        skf = StratifiedKFold(
            n_splits=Config.CV_N_SPLITS, 
            shuffle=True, 
            random_state=Config.RANDOM_SEED
        )
        
        all_fold_results = {}  # {model_name: [fold_results]}
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_paths, all_labels), 1):
            print(f"\n{'='*70}")
            print(f" FOLD {fold_idx}/{Config.CV_N_SPLITS}")
            print(f"{'='*70}")
            
            fold_train_paths = [all_paths[i] for i in train_idx]
            fold_train_labels = [all_labels[i] for i in train_idx]
            fold_test_paths = [all_paths[i] for i in test_idx]
            fold_test_labels = [all_labels[i] for i in test_idx]
            
            # Tách 1 phần từ train làm validation cho early stopping
            from sklearn.model_selection import train_test_split
            fold_train_paths, fold_val_paths, fold_train_labels, fold_val_labels = train_test_split(
                fold_train_paths, fold_train_labels,
                test_size=0.15,
                stratify=fold_train_labels,
                random_state=Config.RANDOM_SEED
            )
            
            print(f"  Train: {len(fold_train_paths)} | Val: {len(fold_val_paths)} | Test (fold): {len(fold_test_paths)}")
            
            # Create dataloaders for this fold
            fold_train_loader, fold_val_loader, fold_test_loader = create_dataloaders(
                fold_train_paths, fold_train_labels,
                fold_val_paths, fold_val_labels,
                fold_test_paths, fold_test_labels,
                Config.BATCH_SIZE,
                Config.NUM_WORKERS
            )
            
            fold_folder = os.path.join(run_folder, f"fold_{fold_idx}")
            os.makedirs(fold_folder, exist_ok=True)
            
            for model_name in Config.MODELS:
                print(f"\n  [Fold {fold_idx}] Training {model_name}...")
                try:
                    checkpoint_manager, history = train_model(
                        model_name,
                        fold_train_loader,
                        fold_val_loader,
                        num_classes,
                        device,
                        class_names=class_names,
                        train_labels=fold_train_labels,
                        save_dir=fold_folder
                    )
                    
                    # Evaluate trên fold test set (phần data luân phiên làm test)
                    strategy_ckpt_dir = os.path.join(fold_folder, model_name, 'checkpoints')
                    results = evaluate_all_strategies(
                        model_name, checkpoint_manager, fold_test_loader, fold_train_loader,
                        num_classes, device, class_names=class_names, save_dir=strategy_ckpt_dir
                    )
                    
                    if model_name not in all_fold_results:
                        all_fold_results[model_name] = []
                    all_fold_results[model_name].append(results)
                    
                    save_model_results(model_name, results, fold_folder)
                    
                    if Config.AUTO_DELETE_CHECKPOINTS:
                        delete_model_checkpoints(model_name, Config.CHECKPOINTS_DIR)
                    
                    print(f"  ✅ Fold {fold_idx} - {model_name} completed")
                    
                except Exception as e:
                    print(f"  ✗ Fold {fold_idx} - {model_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Tổng hợp kết quả CV
        print(f"\n{'='*70}")
        print(f" CROSS-VALIDATION SUMMARY ({Config.CV_N_SPLITS}-Fold)")
        print(f"{'='*70}")
        
        cv_summary_rows = []
        for model_name, fold_results_list in all_fold_results.items():
            # Lấy Strategy 1 (best checkpoint) từ mỗi fold
            for strategy_name in fold_results_list[0].keys():
                fold_metrics = []
                for fold_res in fold_results_list:
                    if strategy_name in fold_res:
                        fold_metrics.append(fold_res[strategy_name]['metrics'])
                
                if fold_metrics:
                    avg_metrics = {}
                    for key in fold_metrics[0].keys():
                        values = [m[key] for m in fold_metrics]
                        avg_metrics[f"{key} (mean)"] = np.mean(values)
                        avg_metrics[f"{key} (std)"] = np.std(values)
                    
                    row = {'Model': model_name, 'Strategy': strategy_name, **avg_metrics}
                    cv_summary_rows.append(row)
        
        cv_df = pd.DataFrame(cv_summary_rows)
        cv_excel_path = os.path.join(run_folder, 'cv_summary_results.xlsx')
        cv_df.to_excel(cv_excel_path, index=False)
        print(f"\n✓ CV Summary saved to: {cv_excel_path}")
        print(cv_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f" CROSS-VALIDATION COMPLETED!")
        print(f"{'='*70}")
        return
    
    # ===================== NORMAL MODE (no CV) =====================
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
                class_names=class_names,
                train_labels=train_labels,
                save_dir=run_folder
            )
            print(f"  ✓ Training completed for {model_name}")
            
            # 3.2: Evaluate with 3 strategies
            print(f"\n  [3.2] Evaluating {model_name} with 3 strategies...")
            # Tạo folder lưu checkpoint cho các strategy
            strategy_checkpoint_dir = os.path.join(run_folder, model_name, 'checkpoints')
            results = evaluate_all_strategies(
                model_name,
                checkpoint_manager,
                test_loader,
                train_loader,  # CRITICAL: Pass train_loader for BatchNorm update
                num_classes,
                device,
                class_names=class_names,
                save_dir=strategy_checkpoint_dir
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
            
            # 3.4: Delete checkpoints to free disk space (conditional)
            if Config.AUTO_DELETE_CHECKPOINTS:
                print(f"\n  [3.4] Cleaning up checkpoints for {model_name}...")
                delete_model_checkpoints(model_name, Config.CHECKPOINTS_DIR)
            else:
                print(f"\n  [3.4] Keeping checkpoints for {model_name} (AUTO_DELETE_CHECKPOINTS=False)")
            
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
    
    # Step 6: Experiment info already saved via export_run_config
    print(f"\n[Step 6/6] Run config already exported at start.")
    
    # Final summary
    print("\n" + "="*70)
    print(" BASELINE RESEARCH COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Lần chạy #{run_number}:")
    print(f"  - Folder: {run_folder}")
    print(f"  - Combined Excel: {excel_path}")
    print(f"  - Combined Chart: {os.path.join(run_folder, 'performance_comparison.png')}")
    print(f"  - Run Config: {os.path.join(run_folder, 'run_config.xlsx')}")
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
