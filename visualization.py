import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _ema_smooth(data, alpha=0.3):
    """
    Exponential Moving Average smoothing.
    alpha closer to 0 → smoother; closer to 1 → noisier (follows raw data).
    """
    smoothed = []
    s = data[0]
    for x in data:
        s = alpha * x + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


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


def plot_dwa_curves(history, save_dir, ema_alpha=0.25):
    """
    Plot DWA analysis with 4 subplots for better visualization:
      1) Individual raw loss curves (log scale)
      2) Lambda weights with EMA smoothing (raw + smoothed lines)
      3) Stacked area chart showing proportion of each lambda
      4) Heatmap of lambda weights over epochs

    Args:
        history (dict): must contain keys raw_ce, raw_l1, raw_l2, raw_l3,
                        raw_dist, lambda_ce, lambda_l1, lambda_l2, lambda_l3,
                        lambda_dist
        save_dir (str): directory to save the output PNG
        ema_alpha (float): EMA smoothing factor (0=very smooth, 1=raw data)
    """
    epochs = list(range(1, len(history["raw_ce"]) + 1))
    n_epochs = len(epochs)

    # Color palette
    colors = {
        'ce':   '#2196F3',  # Blue
        'proj1': '#FF9800', # Orange
        'proj2': '#4CAF50', # Green
        'logits':'#F44336', # Red
        'dist':  '#9C27B0', # Purple
    }

    # Lambda data (only KD losses, excluding CE which is fixed at 1.0)
    lambda_keys = ['lambda_l1', 'lambda_l2', 'lambda_l3', 'lambda_dist']
    lambda_labels = ['Proj1 (PCA)', 'Proj2 (GWL)', 'Logits (KD)', 'DIST']
    lambda_colors = [colors['proj1'], colors['proj2'], colors['logits'], colors['dist']]

    lambda_data = [history[k] for k in lambda_keys]

    # EMA smoothed versions
    lambda_smoothed = [_ema_smooth(d, alpha=ema_alpha) for d in lambda_data]

    # =====================================================================
    # Figure layout: 2x2 subplots
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Dynamic Weight Averaging (DWA) Analysis", fontsize=16, fontweight='bold', y=0.98)

    # ----- Subplot 1: Raw loss curves (log scale) -----
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["raw_ce"],   label="CE Loss",     linewidth=1.5, color=colors['ce'])
    ax1.plot(epochs, history["raw_l1"],   label="Proj1 Loss",  linewidth=1.5, color=colors['proj1'])
    ax1.plot(epochs, history["raw_l2"],   label="Proj2 Loss",  linewidth=1.5, color=colors['proj2'])
    ax1.plot(epochs, history["raw_l3"],   label="Logits Loss", linewidth=1.5, color=colors['logits'])
    ax1.plot(epochs, history["raw_dist"], label="DIST Loss",   linewidth=1.5, color=colors['dist'])
    ax1.set_title("Individual Raw Loss Curves", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Raw Loss (log scale)")
    ax1.set_yscale("log")
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ----- Subplot 2: Lambda weights with EMA smoothing -----
    ax2 = axes[0, 1]
    for i, (label, color) in enumerate(zip(lambda_labels, lambda_colors)):
        # Raw data as thin transparent line
        ax2.plot(epochs, lambda_data[i], color=color, alpha=0.2, linewidth=0.8)
        # Smoothed data as thick solid line
        ax2.plot(epochs, lambda_smoothed[i], color=color, linewidth=2.5,
                 label=f"{label}", marker='', markersize=0)

    # Add horizontal reference line at 1.0 (equal weighting)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label='Equal weight (1.0)')

    ax2.set_title(f"Lambda Weights (EMA smoothed, α={ema_alpha})", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Lambda Weight")
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ----- Subplot 3: Stacked area chart (proportion) -----
    ax3 = axes[1, 0]

    # Normalize lambda to show proportion (sum to 100%)
    lambda_array = np.array(lambda_smoothed)  # shape: [4, n_epochs]
    lambda_sum = lambda_array.sum(axis=0)     # shape: [n_epochs]
    lambda_pct = (lambda_array / lambda_sum) * 100  # percentage

    ax3.stackplot(epochs,
                  lambda_pct[0], lambda_pct[1], lambda_pct[2], lambda_pct[3],
                  labels=lambda_labels,
                  colors=lambda_colors,
                  alpha=0.75)
    ax3.set_title("Lambda Weight Proportions (Stacked Area)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Proportion (%)")
    ax3.set_ylim(0, 100)
    ax3.legend(loc='center right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ----- Subplot 4: Heatmap of lambda weights -----
    ax4 = axes[1, 1]

    # Use smoothed data for heatmap
    heatmap_data = np.array(lambda_smoothed)  # shape: [4, n_epochs]

    im = ax4.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                    interpolation='bilinear')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(lambda_labels, fontsize=10)
    ax4.set_title("Lambda Weights Heatmap", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Epoch")

    # Set x-axis tick labels to show epoch numbers
    if n_epochs <= 30:
        tick_step = 1
    elif n_epochs <= 60:
        tick_step = 5
    elif n_epochs <= 150:
        tick_step = 10
    else:
        tick_step = 20
    tick_positions = list(range(0, n_epochs, tick_step))
    tick_labels = [str(e + 1) for e in tick_positions]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels)

    cbar = fig.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
    cbar.set_label("Lambda Weight", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(save_dir, "dwa_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] DWA curves saved to: {plot_path}")

    # =====================================================================
    # Additional: Individual lambda subplot (each lambda gets its own panel)
    # =====================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle("Individual Lambda Weight Trajectories (DWA)", fontsize=15, fontweight='bold', y=0.98)

    for idx, (ax, label, color) in enumerate(zip(axes2.flat, lambda_labels, lambda_colors)):
        raw = lambda_data[idx]
        smooth = lambda_smoothed[idx]

        # Fill between raw min/max as a band
        ax.fill_between(epochs, raw, alpha=0.15, color=color, label='Raw range')
        ax.plot(epochs, raw, color=color, alpha=0.35, linewidth=0.7, label='Raw')
        ax.plot(epochs, smooth, color=color, linewidth=2.5, label=f'EMA (α={ema_alpha})')

        # Reference line
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

        ax.set_title(f"λ {label}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Lambda Weight")
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path2 = os.path.join(save_dir, "dwa_individual_lambdas.png")
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Individual lambda plots saved to: {plot_path2}")
