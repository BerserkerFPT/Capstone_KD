import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Training curves saved to: {plot_path}")

