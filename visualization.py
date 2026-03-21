"""
visualization.py — Publication-ready DWA visualization suite.

Generates figures for LNCS / IEEE style papers demonstrating
how Dynamic Weight Averaging adjusts task weights based on
relative loss convergence rates.

Output figures:
  1. training_curves.png         — Loss, Accuracy, LR schedule
  2. dwa_overview.png            — Raw losses + smoothed lambdas (2-panel)
  3. dwa_mechanism.png           — Loss ratios r_i(t) → softmax → λ_i(t)
  4. dwa_individual_lambdas.png  — Dual-axis: λ vs Loss for each task
  5. dwa_deviation.png           — Stacked deviation + heatmap (z-score)
  6. dwa_correlation.png         — Scatter: ΔLoss vs Δλ with regression
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch


# =========================================================================
# Shared academic style configuration
# =========================================================================
# Consistent color palette across all figures
COLORS = {
    'proj1':  '#D95F02',   # Orange (ColorBrewer Dark2)
    'proj2':  '#1B9E77',   # Teal
    'logits': '#E7298A',   # Magenta-pink
    'dist':   '#7570B3',   # Blue-violet
    'ce':     '#66A61E',   # Olive-green
}
TASK_KEYS    = ['lambda_ce', 'lambda_l1', 'lambda_l2', 'lambda_l3', 'lambda_dist']
LOSS_KEYS    = ['raw_ce',    'raw_l1',    'raw_l2',    'raw_l3',    'raw_dist']
TASK_LABELS  = [r'$\mathcal{L}_{\mathrm{CE}}$',
                r'$\mathcal{L}_{\mathrm{Proj1}}$',
                r'$\mathcal{L}_{\mathrm{Proj2}}$',
                r'$\mathcal{L}_{\mathrm{Logits}}$',
                r'$\mathcal{L}_{\mathrm{DIST}}$']
TASK_COLORS  = [COLORS['ce'], COLORS['proj1'], COLORS['proj2'],
                COLORS['logits'], COLORS['dist']]

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


def _ema_smooth(data, alpha=0.25):
    """Exponential Moving Average. alpha→0 = smoother."""
    smoothed = []
    s = data[0]
    for x in data:
        s = alpha * x + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


# =========================================================================
# TASK 7: Diagnostics — print statistics and warn if DWA is ineffective
# =========================================================================
def print_dwa_diagnostics(history):
    """
    Print mean/std of loss ratios r_i(t) and lambda weights λ_i(t).
    Warns if std ≈ 0, meaning DWA has negligible effect.
    """
    n = len(history['raw_l1'])
    losses = np.array([history[k] for k in LOSS_KEYS])       # [4, T]
    lambdas = np.array([history[k] for k in TASK_KEYS])       # [4, T]

    # Loss ratios r_i(t) = L_i(t) / L_i(t-1), defined from epoch 2 onwards
    ratios = losses[:, 1:] / (losses[:, :-1] + 1e-8)          # [4, T-1]

    print("\n" + "=" * 72)
    print("📊 DWA DIAGNOSTICS")
    print("=" * 72)
    print(f"{'Task':<20s}  {'mean(r_i)':<12s} {'std(r_i)':<12s} "
          f"{'mean(λ_i)':<12s} {'std(λ_i)':<12s}")
    print("-" * 72)

    any_warning = False
    for i, label in enumerate(TASK_LABELS):
        r_mean = ratios[i].mean()
        r_std  = ratios[i].std()
        l_mean = lambdas[i].mean()
        l_std  = lambdas[i].std()
        tag = ""
        if l_std < 0.01:
            tag = "  ⚠️  std≈0 → DWA negligible"
            any_warning = True
        # Strip LaTeX for printing
        name = label.replace('$', '').replace(r'\mathcal{L}_{\mathrm{', '').replace('}}', '')
        print(f"  {name:<18s}  {r_mean:<12.6f} {r_std:<12.6f} "
              f"{l_mean:<12.6f} {l_std:<12.6f}{tag}")

    # Global statistics
    all_lambda_std = lambdas.std()
    print("-" * 72)
    print(f"  Global λ std across all tasks/epochs: {all_lambda_std:.6f}")
    if all_lambda_std < 0.02:
        print("  ⚠️  WARNING: Overall λ variation is very low. "
              "DWA may be ineffective — consider lowering temperature T.")
        any_warning = True
    if not any_warning:
        print("  ✅ DWA is actively adjusting weights.")
    print("=" * 72 + "\n")


# =========================================================================
# Figure 1: Training curves (unchanged from original)
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


# =========================================================================
# Figure 2: DWA Overview — raw losses + smoothed lambda weights
# =========================================================================
def plot_dwa_overview(history, save_dir, ema_alpha=0.25):
    """Two-panel overview: loss curves and smoothed lambda weights."""
    epochs = list(range(1, len(history["raw_ce"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # --- Panel A: Raw loss curves (log scale) ---
    for i, (key, label, color) in enumerate(zip(LOSS_KEYS, TASK_LABELS, TASK_COLORS)):
        ax1.plot(epochs, history[key], label=label, lw=1.3, color=color)
    _apply_style(ax1, ylabel='Loss (log scale)', title='(a) Individual Loss Curves')
    ax1.set_yscale('log')
    ax1.legend(fontsize=LEGEND_SIZE, framealpha=0.8, ncol=2)

    # --- Panel B: Lambda weights (EMA smoothed) ---
    for i, (key, label, color) in enumerate(zip(TASK_KEYS, TASK_LABELS, TASK_COLORS)):
        raw = history[key]
        smooth = _ema_smooth(raw, alpha=ema_alpha)
        ax2.plot(epochs, raw, color=color, alpha=0.15, lw=0.6)
        ax2.plot(epochs, smooth, color=color, lw=2.0, label=label)
    ax2.axhline(y=1.0, color='#888', ls='--', lw=0.8, alpha=0.5,
                label='Equal weight')
    _apply_style(ax2, ylabel=r'$\lambda_i(t)$',
                 title=f'(b) DWA Weights (EMA $\\alpha$={ema_alpha})')
    ax2.legend(fontsize=LEGEND_SIZE, framealpha=0.8, ncol=2)

    plt.tight_layout()
    path = os.path.join(save_dir, "dwa_overview.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# =========================================================================
# TASK 2: DWA Mechanism — loss ratios r_i → softmax → λ_i
# =========================================================================
def plot_dwa_mechanism(history, save_dir, ema_alpha=0.25):
    """
    Explicitly shows WHY weights change:
      Panel A: Loss ratio r_i(t) = L_i(t-1) / L_i(t-2)
      Panel B: Resulting lambda λ_i(t) = K · softmax(r/T)
      Panel C: Scatter r_i vs λ_i (direct mapping)
    """
    losses  = np.array([history[k] for k in LOSS_KEYS])    # [4, T]
    lambdas = np.array([history[k] for k in TASK_KEYS])    # [4, T]
    T = len(history['raw_l1'])
    epoch_axis = np.arange(3, T + 1)  # ratios start from epoch 3

    # r_i(t) = L_i(t-1) / L_i(t-2) — note: in code, r computed from epoch 3
    ratios = losses[:, 1:] / (losses[:, :-1] + 1e-8)       # [4, T-1]
    # Align: ratio[t-2] corresponds to epoch t (start from epoch 3)
    ratios_plot = ratios[:, 1:]  # [4, T-2], epoch 3..T

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # --- Panel A: Loss ratios r_i(t) ---
    ax = axes[0]
    for i, (label, color) in enumerate(zip(TASK_LABELS, TASK_COLORS)):
        raw_r = ratios_plot[i]
        smooth_r = _ema_smooth(list(raw_r), alpha=ema_alpha)
        ax.plot(epoch_axis, raw_r, color=color, alpha=0.15, lw=0.5)
        ax.plot(epoch_axis, smooth_r, color=color, lw=1.8, label=label)
    ax.axhline(y=1.0, color='#888', ls='--', lw=0.8, alpha=0.5)
    _apply_style(ax, ylabel=r'$r_i(t) = \mathcal{L}_i(t\!-\!1) / \mathcal{L}_i(t\!-\!2)$',
                 title=r'(a) Loss Ratio $r_i(t)$')
    ax.legend(fontsize=LEGEND_SIZE-1, framealpha=0.8)
    # Annotate meaning
    ax.annotate('Slow decrease\n(ratio ≈ 1)', xy=(0.7, 0.85),
                xycoords='axes fraction', fontsize=8, color='#555',
                ha='center', style='italic')
    ax.annotate('Fast decrease\n(ratio < 1)', xy=(0.7, 0.15),
                xycoords='axes fraction', fontsize=8, color='#555',
                ha='center', style='italic')

    # --- Panel B: Resulting lambdas ---
    ax = axes[1]
    lambdas_plot = lambdas[:, 2:]  # align with ratios (epoch 3 onward)
    for i, (label, color) in enumerate(zip(TASK_LABELS, TASK_COLORS)):
        raw_l = lambdas_plot[i]
        smooth_l = _ema_smooth(list(raw_l), alpha=ema_alpha)
        ax.plot(epoch_axis, raw_l, color=color, alpha=0.15, lw=0.5)
        ax.plot(epoch_axis, smooth_l, color=color, lw=1.8, label=label)
    ax.axhline(y=1.0, color='#888', ls='--', lw=0.8, alpha=0.5)
    _apply_style(ax, ylabel=r'$\lambda_i(t) = K \cdot \mathrm{softmax}(r_i/T)$',
                 title=r'(b) DWA Weights $\lambda_i(t)$')
    ax.legend(fontsize=LEGEND_SIZE-1, framealpha=0.8)
    ax.annotate('Higher weight\n(compensate slow loss)',
                xy=(0.65, 0.88), xycoords='axes fraction',
                fontsize=8, color='#555', ha='center', style='italic')
    ax.annotate('Lower weight\n(loss converged)',
                xy=(0.65, 0.12), xycoords='axes fraction',
                fontsize=8, color='#555', ha='center', style='italic')

    # --- Panel C: Direct scatter r_i vs λ_i ---
    ax = axes[2]
    for i, (label, color) in enumerate(zip(TASK_LABELS, TASK_COLORS)):
        r_vals = ratios_plot[i]
        l_vals = lambdas_plot[i]
        ax.scatter(r_vals, l_vals, s=8, alpha=0.35, color=color, label=label)
        # Add smoothed trend as a line
        # Sort by r for a clean curve
        idx = np.argsort(r_vals)
        r_sorted = r_vals[idx]
        l_sorted = l_vals[idx]
        # Bin-average for trend
        n_bins = 20
        if len(r_sorted) > n_bins:
            bin_edges = np.linspace(r_sorted.min(), r_sorted.max(), n_bins + 1)
            r_centers = []
            l_means = []
            for b in range(n_bins):
                mask = (r_sorted >= bin_edges[b]) & (r_sorted < bin_edges[b+1])
                if mask.sum() > 0:
                    r_centers.append(r_sorted[mask].mean())
                    l_means.append(l_sorted[mask].mean())
            ax.plot(r_centers, l_means, color=color, lw=2.0, alpha=0.8)

    ax.axhline(y=1.0, color='#888', ls='--', lw=0.8, alpha=0.4)
    ax.axvline(x=1.0, color='#888', ls='--', lw=0.8, alpha=0.4)
    _apply_style(ax, xlabel=r'Loss ratio $r_i$', ylabel=r'$\lambda_i$',
                 title=r'(c) Mapping: $r_i \rightarrow \lambda_i$')
    ax.legend(fontsize=LEGEND_SIZE-1, framealpha=0.8, markerscale=2)

    plt.tight_layout()
    path = os.path.join(save_dir, "dwa_mechanism.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# =========================================================================
# TASK 1: Individual lambda + loss dual-axis plots
# =========================================================================
def plot_dwa_individual(history, save_dir, ema_alpha=0.25):
    """
    For each loss (including CE): dual-axis subplot
      Left Y: λ_i (raw transparent + EMA solid)
      Right Y: Loss L_i (dashed)
    Shows how lambda reacts to loss dynamics.
    """
    epochs = list(range(1, len(history['raw_ce']) + 1))
    num_tasks = len(TASK_KEYS)

    # Layout: 3 on top row, 2 centered on bottom row
    fig = plt.figure(figsize=(18, 10))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.8)

    ax_positions = [
        gs[0, 0:2],   # top-left
        gs[0, 2:4],   # top-center
        gs[0, 4:6],   # top-right
        gs[1, 1:3],   # bottom-left (centered)
        gs[1, 3:5],   # bottom-right (centered)
    ]

    for idx in range(num_tasks):
        ax = fig.add_subplot(ax_positions[idx])
        lam_key  = TASK_KEYS[idx]
        loss_key = LOSS_KEYS[idx]
        label    = TASK_LABELS[idx]
        color    = TASK_COLORS[idx]
        loss_color = '#555555'

        lam_raw    = history[lam_key]
        lam_smooth = _ema_smooth(lam_raw, alpha=ema_alpha)
        loss_raw   = history[loss_key]
        loss_smooth = _ema_smooth(loss_raw, alpha=ema_alpha)

        # Left axis: lambda
        ax.plot(epochs, lam_raw, color=color, alpha=0.2, lw=0.6)
        ax.plot(epochs, lam_smooth, color=color, lw=2.2,
                label=f'{label} (λ, smoothed)')
        ax.axhline(y=1.0, color='#aaa', ls='--', lw=0.7, alpha=0.5)
        ax.set_ylabel(r'$\lambda_i(t)$', fontsize=LABEL_SIZE, color=color)
        ax.tick_params(axis='y', labelcolor=color, labelsize=TICK_SIZE)

        # Right axis: loss
        ax2 = ax.twinx()
        ax2.plot(epochs, loss_raw, color=loss_color, alpha=0.15, lw=0.5)
        ax2.plot(epochs, loss_smooth, color=loss_color, lw=1.8, ls='--',
                 label=f'{label} (loss)')
        ax2.set_ylabel('Loss', fontsize=LABEL_SIZE, color=loss_color)
        ax2.tick_params(axis='y', labelcolor=loss_color, labelsize=TICK_SIZE)
        ax2.spines['top'].set_visible(False)

        ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
        ax.set_title(f'{label}', fontsize=TITLE_SIZE, fontweight='bold', pad=6)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.2, lw=0.5)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=LEGEND_SIZE - 1, loc='best', framealpha=0.8)

    fig.suptitle(r'Individual $\lambda_i$ vs Loss (Dual-Axis)',
                 fontsize=14, fontweight='bold', y=1.01)
    path = os.path.join(save_dir, "dwa_individual_lambdas.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] {path}")


# =========================================================================
# TASK 3 + TASK 5 + TASK 6: Deviation stacked area + z-score heatmap
# =========================================================================
def plot_dwa_deviation(history, save_dir, ema_alpha=0.25):
    """
    Panel A: Stacked area of (λ_i - mean) showing meaningful variation
    Panel B: Z-score normalized heatmap with high-variance highlights
    """
    epochs = list(range(1, len(history['raw_l1']) + 1))
    n = len(epochs)
    lambdas = np.array([history[k] for k in TASK_KEYS])       # [4, T]

    # Smooth for cleaner visualization
    lam_smooth = np.array([_ema_smooth(list(lambdas[i]), alpha=ema_alpha)
                           for i in range(5)])                   # [5, T]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Panel A: Deviation from mean (stacked area) ---
    ax = axes[0]
    lam_mean = lam_smooth.mean(axis=0)  # mean across tasks at each epoch
    deviations = lam_smooth - lam_mean  # [4, T]

    # Split positive and negative for stacked visualization
    for i, (label, color) in enumerate(zip(TASK_LABELS, TASK_COLORS)):
        ax.fill_between(epochs, 0, deviations[i], alpha=0.35, color=color)
        ax.plot(epochs, deviations[i], color=color, lw=1.5, label=label)
    ax.axhline(y=0, color='#888', ls='-', lw=0.8, alpha=0.5)

    # TASK 6: Highlight high-variance regions
    per_epoch_std = lam_smooth.std(axis=0)  # [T]
    threshold = np.percentile(per_epoch_std, 75)
    high_var_mask = per_epoch_std > threshold
    for i in range(n):
        if high_var_mask[i]:
            ax.axvspan(epochs[i] - 0.5, epochs[i] + 0.5,
                       alpha=0.08, color='red', zorder=0)

    _apply_style(ax, ylabel=r'$\lambda_i - \bar{\lambda}$',
                 title=r'(a) Weight Deviation from Mean')
    ax.legend(fontsize=LEGEND_SIZE - 1, framealpha=0.8, ncol=2)
    # Add annotation for highlighted regions
    ax.annotate('Shaded: high variance epochs',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=7, color='red', alpha=0.7, ha='right', style='italic')

    # --- Panel B: Z-score normalized heatmap ---
    ax = axes[1]
    # Z-score: per-task normalization
    z_scores = np.zeros_like(lam_smooth)
    for i in range(5):
        mu = lam_smooth[i].mean()
        sigma = lam_smooth[i].std()
        if sigma > 1e-8:
            z_scores[i] = (lam_smooth[i] - mu) / sigma
        else:
            z_scores[i] = 0.0

    im = ax.imshow(z_scores, aspect='auto', cmap='RdBu_r',
                   interpolation='bilinear', vmin=-2.5, vmax=2.5)

    # Y-axis labels
    short_labels = ['CE', 'Proj1', 'Proj2', 'Logits', 'DIST']
    ax.set_yticks(range(5))
    ax.set_yticklabels(short_labels, fontsize=TICK_SIZE)

    # X-axis ticks
    if n <= 30:
        step = 2
    elif n <= 60:
        step = 5
    elif n <= 150:
        step = 10
    else:
        step = 20
    positions = list(range(0, n, step))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p + 1) for p in positions], fontsize=TICK_SIZE)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Z-score', fontsize=LABEL_SIZE - 1)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    # TASK 6: Mark high-variance columns
    for i in range(n):
        if high_var_mask[i]:
            ax.axvline(x=i, color='red', alpha=0.12, lw=1.5)

    ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
    ax.set_title(r'(b) $\lambda$ Heatmap (Z-score normalized)', fontsize=TITLE_SIZE,
                 fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "dwa_deviation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# =========================================================================
# TASK 4: Correlation — ΔLoss vs Δλ
# =========================================================================
def plot_dwa_correlation(history, save_dir):
    """
    Scatter plot: ΔLoss_i(t) vs Δλ_i(t) with correlation coefficient.
    Demonstrates core DWA property:
      Faster decreasing loss (ΔL < 0) → weight decreases (Δλ < 0)
      Slower decreasing loss (ΔL ≈ 0) → weight increases (Δλ > 0)
    """
    losses  = np.array([history[k] for k in LOSS_KEYS])    # [4, T]
    lambdas = np.array([history[k] for k in TASK_KEYS])    # [4, T]

    # Compute changes (finite differences)
    delta_loss   = np.diff(losses,  axis=1)    # [4, T-1]
    delta_lambda = np.diff(lambdas, axis=1)    # [4, T-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel A: Per-task scatter ---
    ax = axes[0]
    all_dl = []
    all_dlam = []
    for i, (label, color) in enumerate(zip(TASK_LABELS, TASK_COLORS)):
        dl   = delta_loss[i]
        dlam = delta_lambda[i]
        ax.scatter(dl, dlam, s=12, alpha=0.4, color=color, label=label,
                   edgecolors='none')
        all_dl.extend(dl)
        all_dlam.extend(dlam)

    # Global regression line
    all_dl   = np.array(all_dl)
    all_dlam = np.array(all_dlam)
    if len(all_dl) > 2:
        corr = np.corrcoef(all_dl, all_dlam)[0, 1]
        # Fit line
        z = np.polyfit(all_dl, all_dlam, 1)
        p = np.poly1d(z)
        x_range = np.linspace(all_dl.min(), all_dl.max(), 100)
        ax.plot(x_range, p(x_range), color='black', lw=1.5, ls='--',
                alpha=0.6, label=f'r = {corr:.3f}')
    else:
        corr = 0.0

    ax.axhline(y=0, color='#888', ls='-', lw=0.5, alpha=0.4)
    ax.axvline(x=0, color='#888', ls='-', lw=0.5, alpha=0.4)
    _apply_style(ax, xlabel=r'$\Delta\mathcal{L}_i$ (loss change)',
                 ylabel=r'$\Delta\lambda_i$ (weight change)',
                 title=r'(a) $\Delta\mathcal{L}$ vs $\Delta\lambda$ (all tasks)')
    ax.legend(fontsize=LEGEND_SIZE - 1, framealpha=0.8, markerscale=2)

    # --- Panel B: Per-task correlation bars ---
    ax = axes[1]
    correlations = []
    for i in range(5):
        dl   = delta_loss[i]
        dlam = delta_lambda[i]
        if len(dl) > 2 and np.std(dl) > 1e-10 and np.std(dlam) > 1e-10:
            r = np.corrcoef(dl, dlam)[0, 1]
        else:
            r = 0.0
        correlations.append(r)

    short_labels = ['CE', 'Proj1', 'Proj2', 'Logits', 'DIST']
    bar_colors = [c if r > 0 else '#999' for c, r in zip(TASK_COLORS, correlations)]
    bars = ax.bar(range(5), correlations, color=TASK_COLORS, alpha=0.75,
                  edgecolor='white', linewidth=0.8)

    # Add value labels
    for bar, r in zip(bars, correlations):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.01 * np.sign(y),
                f'{r:.3f}', ha='center', va='bottom' if y >= 0 else 'top',
                fontsize=TICK_SIZE, fontweight='bold')

    ax.set_xticks(range(5))
    ax.set_xticklabels(short_labels, fontsize=TICK_SIZE)
    ax.axhline(y=0, color='#888', ls='-', lw=0.8, alpha=0.5)
    _apply_style(ax, xlabel='Task', ylabel='Pearson r',
                 title=r'(b) Correlation $\Delta\mathcal{L}$ vs $\Delta\lambda$')

    # Interpretation annotation
    interpretation = ("Positive r: weight increases when loss increases\n"
                      "→ DWA compensates for non-converging losses")
    ax.annotate(interpretation, xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=7, color='#555', ha='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                          edgecolor='#ccc', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "dwa_correlation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# =========================================================================
# Master function: generate all DWA plots
# =========================================================================
def plot_dwa_curves(history, save_dir, ema_alpha=0.25):
    """
    Generate ALL DWA visualization figures (Tasks 1-7).
    Called from main.py after training.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Task 7: Print diagnostics first
    print_dwa_diagnostics(history)

    # Task 2: DWA mechanism (loss ratios → lambdas)
    plot_dwa_mechanism(history, save_dir, ema_alpha)

    # Overview: raw losses + smoothed lambdas
    plot_dwa_overview(history, save_dir, ema_alpha)

    # Task 1: Individual dual-axis plots
    plot_dwa_individual(history, save_dir, ema_alpha)

    # Task 3 + 5 + 6: Deviation + heatmap + highlights
    plot_dwa_deviation(history, save_dir, ema_alpha)

    # Task 4: Correlation scatter
    plot_dwa_correlation(history, save_dir)

    print(f"\n[OK] All DWA figures saved to: {save_dir}/")
