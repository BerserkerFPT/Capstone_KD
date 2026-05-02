"""
AgriKD — Baseline Model Selection Pipeline

Benchmarks candidate teacher and student architectures on a given dataset.
Run this file to train all models listed in Config.MODELS, evaluate them
with three checkpoint strategies, and export results to Excel.
"""
import os
import shutil

# Force CPU mode if GPU has memory issues
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Uncomment this line to force CPU mode
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    """Return a new numbered subfolder under base_results_dir (1, 2, 3, …)."""
    os.makedirs(base_results_dir, exist_ok=True)

    existing_runs = [
        int(item)
        for item in os.listdir(base_results_dir)
        if os.path.isdir(os.path.join(base_results_dir, item)) and item.isdigit()
    ]
    next_run = max(existing_runs) + 1 if existing_runs else 1
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
    """Export full run configuration to run_config.xlsx (one sheet per group)."""
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
    Full pipeline:
      1. Validate config
      2. Load and split dataset
      3. Train each model, evaluate with 3 strategies, save results
      4. Combine all results to Excel + performance charts
    """
    print("\n" + "="*70)
    print("  AgriKD — BASELINE MODEL SELECTION")
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
    
    # Create run-specific output folder
    run_folder, run_number = get_next_run_folder(Config.RESULTS_DIR)
    print(f"\n📁 Run #{run_number} → {run_folder}")

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
        print("\n⚠ Both WeightedRandomSampler and PolyFocalLoss are active.")
        print("  WRS rebalances at the data level; Focal Loss at the loss level.")
        print("  This may double-correct for class imbalance — consider disabling one.")
    
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
            
            # Split fold train into train + val for early stopping
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
                    fold_ckpt_dir = os.path.join(fold_folder, model_name, 'training_checkpoints')
                    checkpoint_manager, history = train_model(
                        model_name,
                        fold_train_loader,
                        fold_val_loader,
                        num_classes,
                        device,
                        class_names=class_names,
                        train_labels=fold_train_labels,
                        save_dir=fold_folder,
                        checkpoints_dir=fold_ckpt_dir
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

                    # In kết quả tất cả strategies của fold này
                    print(f"\n  📊 Fold {fold_idx} - {model_name}:")
                    for strategy_name, result in results.items():
                        m = result['metrics']
                        print(f"     {strategy_name:<30} Acc: {m['Accuracy (%)']:.2f}% | F1: {m['F1-Score (%)']:.2f}% | AUC: {m['AUC (%)']:.2f}%")

                    save_model_results(model_name, results, fold_folder)

                    # Keep best checkpoint for this fold
                    if os.path.exists(strategy_ckpt_dir):
                        ckpt_src = os.path.join(strategy_ckpt_dir, 'best_checkpoint_eval.pth')
                        ckpt_dst = os.path.join(strategy_ckpt_dir, f'best_fold{fold_idx}.pth')
                        if os.path.exists(ckpt_src):
                            os.rename(ckpt_src, ckpt_dst)
                            print(f"    ✓ Kept: best checkpoint → best_fold{fold_idx}.pth")

                    # Delete training checkpoints (epoch_*.pth, best_checkpoint.pth, etc.)
                    if os.path.exists(fold_ckpt_dir):
                        try:
                            shutil.rmtree(fold_ckpt_dir)
                            print(f"    🧹 Deleted training checkpoints: {fold_ckpt_dir}")
                        except Exception as e:
                            print(f"    ⚠ Could not delete training checkpoints: {e}")
                    
                    print(f"  ✅ Fold {fold_idx} - {model_name} completed")
                    
                except Exception as e:
                    print(f"  ✗ Fold {fold_idx} - {model_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # ===== Cross-Validation Summary =====
        print(f"\n{'='*70}")
        print(f" CROSS-VALIDATION SUMMARY ({Config.CV_N_SPLITS}-Fold)")
        print(f"{'='*70}")

        metric_keys = ['Test Loss', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']

        # === Sheet 1: CV Summary — Mean ± Std per model ===
        cv_summary_rows = []
        eval_key = 'Best Checkpoint'
        for model_name, fold_results_list in all_fold_results.items():
            fold_metrics = [
                fold_res[eval_key]['metrics']
                for fold_res in fold_results_list
                if eval_key in fold_res
            ]
            if fold_metrics:
                row = {'Model': model_name}
                for key in metric_keys:
                    values = [m[key] for m in fold_metrics if key in m]
                    if values:
                        mean_v = np.mean(values)
                        std_v  = np.std(values)
                        fmt = '.4f' if key == 'Test Loss' else '.2f'
                        row[f"{key} (mean)"] = round(mean_v, 4 if key == 'Test Loss' else 2)
                        row[f"{key} (std)"]  = round(std_v,  4 if key == 'Test Loss' else 2)
                        row[f"{key} (mean ± std)"] = f"{mean_v:{fmt}} ± {std_v:{fmt}}"
                cv_summary_rows.append(row)

        cv_summary_df = pd.DataFrame(cv_summary_rows)

        # === Sheet 2: Fold Details — raw per-fold values ===
        cv_detail_rows = []
        for model_name, fold_results_list in all_fold_results.items():
            for fold_idx_0, fold_res in enumerate(fold_results_list):
                if eval_key in fold_res:
                    m = fold_res[eval_key]['metrics']
                    detail_row = {'Model': model_name, 'Fold': fold_idx_0 + 1}
                    for key in metric_keys:
                        if key in m:
                            detail_row[key] = round(m[key], 4 if key == 'Test Loss' else 2)
                    cv_detail_rows.append(detail_row)

        cv_detail_df = pd.DataFrame(cv_detail_rows)
        if not cv_detail_df.empty:
            cv_detail_df = cv_detail_df.sort_values(['Model', 'Fold', 'Strategy']).reset_index(drop=True)

        # Export to Excel
        cv_excel_path = os.path.join(run_folder, 'cv_summary_results.xlsx')
        with pd.ExcelWriter(cv_excel_path, engine='openpyxl') as writer:
            cv_summary_df.to_excel(writer, sheet_name='CV Summary', index=False)
            cv_detail_df.to_excel(writer, sheet_name='Fold Details', index=False)

        print(f"\n✓ CV Summary saved to: {cv_excel_path}")
        print(f"  - Sheet 'CV Summary': Mean ± Std per strategy across {Config.CV_N_SPLITS} folds")
        print(f"  - Sheet 'Fold Details': Raw values per fold per strategy")
        
        print(f"\n{'='*70}")
        print(f" CV SUMMARY (Mean ± Std):")
        print(f"{'='*70}")
        for _, row in cv_summary_df.iterrows():
            print(f"  {row['Model']}:")
            for key in metric_keys:
                col = f"{key} (mean ± std)"
                if col in row:
                    print(f"    {key}: {row[col]}")
        
        print(f"\n{'='*70}")
        print(f" CROSS-VALIDATION COMPLETED!")
        print(f"{'='*70}")
        return
    
    # ===== Normal Mode (no CV) =====
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
            
            # 3.2: Evaluate using best checkpoint
            print(f"\n  [3.2] Evaluating {model_name}...")
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
    
    # Find best model (highest F1-Score, best checkpoint)
    best_df = df[df['Strategy'] == 'Best Checkpoint']
    best_idx = best_df['F1-Score (%)'].idxmax()
    best_model = best_df.loc[best_idx, 'Model']
    best_f1   = best_df.loc[best_idx, 'F1-Score (%)']
    best_acc  = best_df.loc[best_idx, 'Accuracy (%)']

    print(f"\n🏆 Best Model:")
    print(f"  Model: {best_model}")
    print(f"  Accuracy: {best_acc:.2f}%")
    print(f"  F1-Score: {best_f1:.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
