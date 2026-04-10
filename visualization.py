"""
visualization.py â€” Training curve visualization.

Output figures:
  1. training_curves.png  â€” Loss, Accuracy, LR schedule
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Font sizes for publication
TITLE_SIZE   = 13
LABEL_SIZE   = 11
TICK_SIZE    = 9
LEGEND_SIZE  = 9

def _apply_style(ax, xlabel='Epoch', ylabel=None, title=None):
    """Apply clean academic styling to an axes."""
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=8)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5)


# =========================================================================
# Figure 1: Training curves
# =========================================================================
def plot_training_curves(history, save_dir):
    """Training loss, accuracy, and LR schedule (3 panels)."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", lw=1.4)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   lw=1.4)
    _apply_style(axes[0], ylabel='Loss', title='Loss Curve')
    axes[0].legend(fontsize=LEGEND_SIZE, framealpha=0.8)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", lw=1.4)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   lw=1.4)
    _apply_style(axes[1], ylabel='Accuracy (%)', title='Accuracy Curve')
    axes[1].legend(fontsize=LEGEND_SIZE, framealpha=0.8)

    # LR schedule
    axes[2].plot(epochs, history["lr"], color='#E67E22', lw=1.4)
    _apply_style(axes[2], ylabel='Learning Rate', title='LR Schedule')
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

