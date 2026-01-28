"""
Main script to run complete baseline research pipeline
"""
import os
import shutil
import argparse
import json
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime

from config import Config
from dataset import load_dataset, create_dataloaders
from train import train_model
from evaluate import evaluate_all_strategies, export_results_to_excel, create_performance_charts
from visualization import print_dataset_statistics
from save import ExperimentTracker


# ============================================================
# Utils
# ============================================================
def get_next_run_folder(base_results_dir):
    os.makedirs(base_results_dir, exist_ok=True)
    runs = [int(d) for d in os.listdir(base_results_dir)
            if d.isdigit() and os.path.isdir(os.path.join(base_results_dir, d))]
    run_number = max(runs) + 1 if runs else 1
    run_folder = os.path.join(base_results_dir, str(run_number))
    os.makedirs(run_folder, exist_ok=True)
    return run_folder, run_number


def save_model_results(model_name, results, output_dir):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    rows = []
    for strategy, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Strategy": strategy,
            **metrics
        })

    df = pd.DataFrame(rows)
    df = df[
        ['Model', 'Strategy', 'Test Loss',
         'Accuracy (%)', 'Precision (%)',
         'Recall (%)', 'F1-Score (%)', 'AUC (%)']
    ]

    path = os.path.join(model_dir, f"{model_name}_results.xlsx")
    df.to_excel(path, index=False)
    print(f"  ✓ Saved results to {path}")
    return df


def delete_model_checkpoints(model_name, ckpt_dir):
    path = os.path.join(ckpt_dir, model_name)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"  ✓ Deleted checkpoints: {path}")


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser("Baseline Research Pipeline")

    # Dataset split
    parser.add_argument("--train_ratio", type=float, default=Config.TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=Config.VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=Config.TEST_RATIO)

    # Training
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED)

    # Optimization
    parser.add_argument("--early_stop", type=int, default=Config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--lr_decay_patience", type=int, default=Config.LR_DECAY_PATIENCE)
    parser.add_argument("--lr_decay_factor", type=float, default=Config.LR_DECAY_FACTOR)

    # Scheduler
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--eta_min", type=float, default=Config.ETA_MIN)

    # Model
    parser.add_argument("--models", nargs="+", default=Config.MODELS)
    parser.add_argument("--dropout", type=float, default=Config.DROPOUT_RATE)

    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--exp_name", type=str, default=Config.EXPERIMENT_NAME)

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "=" * 70)
    print(" BASELINE RESEARCH - PRETRAINED MODELS EVALUATION")
    print("=" * 70)

    # ========================================================
    # Seed (AFTER override Config)
    # ========================================================
    print(f"\n🔒 Setting random seed = {Config.RANDOM_SEED}")
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("✓ Seed set")

    # ========================================================
    # Validate config
    # ========================================================
    Config.validate_config()

    print(f"🔥 Warmup epochs: {Config.WARMUP_EPOCHS}/{Config.NUM_EPOCHS}")
    print(f"📉 Cosine eta_min: {Config.ETA_MIN}")

    # ========================================================
    # Initialize Experiment Tracker
    # ========================================================
    tracker = ExperimentTracker()
    experiment_number = tracker.save_experiment_config(
        experiment_name=Config.EXPERIMENT_NAME
    )

    # ========================================================
    # Run folder
    # ========================================================
    run_folder, run_number = get_next_run_folder(Config.RESULTS_DIR)
    print(f"\n📁 Run #{run_number}: {run_folder}")

    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================================================
    # Load dataset
    # ========================================================
    print("\n[1/6] Loading dataset...")
    train_p, train_l, val_p, val_l, test_p, test_l, class_names = load_dataset(
        Config.DATASET_PATH,
        Config.TRAIN_RATIO,
        Config.VAL_RATIO,
        Config.TEST_RATIO,
        Config.RANDOM_SEED
    )

    num_classes = len(class_names)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_p, train_l,
        val_p, val_l,
        test_p, test_l,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )

    print_dataset_statistics(
        train_p + val_p + test_p,
        train_l + val_l + test_l,
        class_names
    )

    # ========================================================
    # Train + Evaluate
    # ========================================================
    all_results = {}
    success_models = []

    for idx, model_name in enumerate(Config.MODELS, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(Config.MODELS)}] {model_name}")
        print(f"{'='*70}")

        try:
            ckpt_manager, _ = train_model(
                model_name,
                train_loader,
                val_loader,
                num_classes,
                device,
                class_names=class_names
            )

            results = evaluate_all_strategies(
                model_name,
                ckpt_manager,
                test_loader,
                train_loader,
                num_classes,
                device,
                save_dir=os.path.join(run_folder, model_name)
            )

            all_results[model_name] = results
            save_model_results(model_name, results, run_folder)
            
            # Update experiment tracker with results
            # Lấy kết quả từ strategy tốt nhất (ví dụ: best_checkpoint)
            best_strategy_results = results.get('best_checkpoint', {})
            tracker.update_experiment_results(
                experiment_number=experiment_number,
                model_name=model_name,
                results={
                    'best_val_acc': best_strategy_results.get('Accuracy (%)', 'N/A'),
                    'best_val_loss': best_strategy_results.get('Test Loss', 'N/A'),
                    'test_acc': best_strategy_results.get('Accuracy (%)', 'N/A'),
                    'test_loss': best_strategy_results.get('Test Loss', 'N/A'),
                    'precision': best_strategy_results.get('Precision (%)', 'N/A'),
                    'recall': best_strategy_results.get('Recall (%)', 'N/A'),
                    'f1_score': best_strategy_results.get('F1-Score (%)', 'N/A'),
                    'auc': best_strategy_results.get('AUC (%)', 'N/A')
                }
            )
            
            delete_model_checkpoints(model_name, Config.CHECKPOINTS_DIR)
            success_models.append(model_name)

        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
            continue

    # ========================================================
    # Export results
    # ========================================================
    print("\n[5/6] Exporting results...")
    excel_path = os.path.join(run_folder, "all_models_results.xlsx")
    df = export_results_to_excel(all_results, excel_path)
    create_performance_charts(df, run_folder)

    # ========================================================
    # Save metadata
    # ========================================================
    print("\n[6/6] Saving metadata...")
    info = {
        "run_number": run_number,
        "models": success_models,
        "num_classes": num_classes,
        "config": {
            "epochs": Config.NUM_EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "lr": Config.LEARNING_RATE,
            "warmup_epochs": Config.WARMUP_EPOCHS,
            "eta_min": Config.ETA_MIN
        }
    }

    with open(os.path.join(run_folder, "experiment_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # ========================================================
    # Mark experiment as completed and generate summary
    # ========================================================
    tracker.mark_experiment_completed(experiment_number)
    summary_df = tracker.generate_summary_table()
    
    # Print experiment summary
    print(f"\n📊 Experiment #{experiment_number} Summary:")
    tracker.print_experiment_info(experiment_number)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    args = parse_args()

    # Dataset
    Config.TRAIN_RATIO = args.train_ratio
    Config.VAL_RATIO = args.val_ratio
    Config.TEST_RATIO = args.test_ratio

    # Training
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    Config.WEIGHT_DECAY = args.weight_decay
    Config.NUM_WORKERS = args.num_workers
    Config.RANDOM_SEED = args.seed

    # Optimization
    Config.EARLY_STOPPING_PATIENCE = args.early_stop
    Config.LR_DECAY_PATIENCE = args.lr_decay_patience
    Config.LR_DECAY_FACTOR = args.lr_decay_factor

    # Scheduler
    Config.WARMUP_EPOCHS = int(Config.NUM_EPOCHS * args.warmup_ratio)
    Config.ETA_MIN = args.eta_min

    # Model
    Config.MODELS = args.models
    Config.DROPOUT_RATE = args.dropout

    # WandB
    Config.USE_WANDB = args.wandb
    Config.EXPERIMENT_NAME = args.exp_name

    main()
