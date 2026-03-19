import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curves(history, save_dir):
    """
    Plot and save learning curves + LR schedule after training.

    Args:
        history (dict): keys: train_loss, val_loss, train_acc, val_acc, lr
        save_dir (str): directory to save the output PNG
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Loss curve ---
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=1.5)
    axes[0].set_title("Learning Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Accuracy curve ---
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   linewidth=1.5)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Learning rate schedule ---
    axes[2].plot(epochs, history["lr"], color="orange", linewidth=1.5)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")
    
    # Hiển thị nhiều mốc giá trị hơn ở trục Y (bằng cả major lẫn minor ticks)
    axes[2].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    axes[2].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=10))
    axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    axes[2].grid(True, which="major", linestyle='-', alpha=0.5)
    axes[2].grid(True, which="minor", linestyle=':', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Training curves saved to: {plot_path}")

def plot_dwa_curves(history, save_dir):
    """
    Plot and save DWA individual loss curves and lambda weights.
    """
    epochs = list(range(1, len(history["raw_ce"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: 5 raw losses ---
    axes[0].plot(epochs, history["raw_ce"],   label="CE Task Loss", linewidth=1.5, color='blue')
    axes[0].plot(epochs, history["raw_l1"],   label="Proj1 Loss",   linewidth=1.5, color='orange')
    axes[0].plot(epochs, history["raw_l2"],   label="Proj2 Loss",   linewidth=1.5, color='green')
    axes[0].plot(epochs, history["raw_l3"],   label="Logits Loss",  linewidth=1.5, color='red')
    axes[0].plot(epochs, history["raw_dist"], label="DIST Loss",    linewidth=1.5, color='purple')
    axes[0].set_title("Individual Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Raw Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: 5 lambdas ---
    axes[1].plot(epochs, history["lambda_ce"],   label="Lambda CE",    linewidth=1.5, color='blue')
    axes[1].plot(epochs, history["lambda_l1"],   label="Lambda Proj1", linewidth=1.5, color='orange')
    axes[1].plot(epochs, history["lambda_l2"],   label="Lambda Proj2", linewidth=1.5, color='green')
    axes[1].plot(epochs, history["lambda_l3"],   label="Lambda Logits",linewidth=1.5, color='red')
    axes[1].plot(epochs, history["lambda_dist"], label="Lambda DIST",  linewidth=1.5, color='purple')
    axes[1].set_title("DWA Lambda Weights Evolution")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Lambda Weight")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "dwa_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] DWA curves saved to: {plot_path}")

