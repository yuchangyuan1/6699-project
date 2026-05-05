"""Plotting functions for all 8 required plots + LR search plot."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global style
SGD_COLOR = '#E07B54'
ADAM_COLOR = '#5B8DB8'
OPT_COLORS = {'sgd': SGD_COLOR, 'adam': ADAM_COLOR}
BAND_ALPHA = 0.2
DPI = 150

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': DPI,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + '.png', bbox_inches='tight', dpi=DPI)
    fig.savefig(str(path) + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_loss_vs_epoch(agg_epoch: dict, title: str, save_path):
    """Loss vs epoch for all optimizers. agg_epoch: {opt_name: DataFrame}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, df in agg_epoch.items():
        color = OPT_COLORS.get(opt, 'gray')
        ax.plot(df['epoch'], df['mean_loss'], label=opt.upper(), color=color, linewidth=2)
        ax.fill_between(
            df['epoch'],
            df['mean_loss'] - df['std_loss'],
            df['mean_loss'] + df['std_loss'],
            alpha=BAND_ALPHA, color=color,
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_loss_vs_step(agg_step: dict, title: str, save_path):
    """Loss vs global step (smoothed) for all optimizers."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for opt, df in agg_step.items():
        color = OPT_COLORS.get(opt, 'gray')
        # Raw (light)
        ax.plot(df['global_step'], df['mean_loss'], alpha=0.15, color=color, linewidth=0.5)
        # Smoothed
        ax.plot(df['global_step'], df['mean_loss_smooth'],
                label=opt.upper(), color=color, linewidth=1.8)
        ax.fill_between(
            df['global_step'],
            df['mean_loss_smooth'] - df['std_loss'],
            df['mean_loss_smooth'] + df['std_loss'],
            alpha=BAND_ALPHA, color=color,
        )
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Training Loss')
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_early_stage_zoom(agg_epoch: dict, title: str, save_path, zoom_epochs: int = 5):
    """Zoom into the first zoom_epochs epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for opt, df in agg_epoch.items():
        color = OPT_COLORS.get(opt, 'gray')
        sub = df[df['epoch'] <= zoom_epochs]
        ax.plot(sub['epoch'], sub['mean_loss'], label=opt.upper(), color=color, linewidth=2, marker='o', markersize=4)
        ax.fill_between(
            sub['epoch'],
            sub['mean_loss'] - sub['std_loss'],
            sub['mean_loss'] + sub['std_loss'],
            alpha=BAND_ALPHA, color=color,
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_threshold_comparison(agg_thresh: dict, title: str, save_path):
    """Grouped bar chart: steps to reach each loss threshold."""
    thresholds = sorted(set().union(*[set(df['threshold']) for df in agg_thresh.values()]), reverse=True)
    opts = list(agg_thresh.keys())
    n_thresh = len(thresholds)
    n_opts = len(opts)
    x = np.arange(n_thresh)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, opt in enumerate(opts):
        color = OPT_COLORS.get(opt, 'gray')
        df = agg_thresh[opt].set_index('threshold')
        means, stds = [], []
        for t in thresholds:
            if t in df.index:
                m = df.loc[t, 'mean_first_global_step']
                s = df.loc[t, 'std_first_global_step']
            else:
                m, s = None, 0
            means.append(m)
            stds.append(s if s and not np.isnan(s) else 0)

        offset = (i - (n_opts - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            [m if m and not np.isnan(m) else 0 for m in means],
            width,
            label=opt.upper(),
            color=color,
            alpha=0.85,
            yerr=stds,
            capsize=4,
        )
        # Hatching for unreached thresholds
        for j, (bar, m) in enumerate(zip(bars, means)):
            if m is None or (isinstance(m, float) and np.isnan(m)):
                bar.set_hatch('///')
                bar.set_edgecolor('gray')
                ax.text(bar.get_x() + bar.get_width() / 2, 50, 'N/A',
                        ha='center', va='bottom', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([f'≤ {t}' for t in thresholds])
    ax.set_xlabel('Loss Threshold')
    ax.set_ylabel('Global Step (first crossing)')
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_lr_search(lr_search_results: dict, save_path):
    """
    lr_search_results: {
        'sgd': {lr: epoch_df, ...},
        'adam': {lr: epoch_df, ...},
    }
    """
    n_opts = len(lr_search_results)
    fig, axes = plt.subplots(1, n_opts, figsize=(6 * n_opts, 4.5), sharey=False)
    if n_opts == 1:
        axes = [axes]

    cmaps = {'sgd': plt.cm.Reds, 'adam': plt.cm.Blues}
    for ax, (opt, lr_dict) in zip(axes, lr_search_results.items()):
        lrs = list(lr_dict.keys())
        cmap = cmaps.get(opt, plt.cm.viridis)
        colors = [cmap(0.3 + 0.6 * i / max(len(lrs) - 1, 1)) for i in range(len(lrs))]
        for lr, color in zip(lrs, colors):
            df = lr_dict[lr]
            ax.plot(df['epoch'], df['loss'], label=f'lr={lr:.0e}', color=color, linewidth=1.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'{opt.upper()} — LR Search')
        ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    _save(fig, save_path)


def generate_all_plots(cfg: dict, agg_results: dict, output_dir: Path):
    """
    Generate all 8 required plots for one experiment config.
    agg_results: output of build_aggregated_outputs().
    """
    exp_name = cfg['experiment']['name']
    model = cfg['experiment']['model']
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    agg_epoch = agg_results['agg_epoch']
    agg_step = agg_results['agg_step']
    agg_thresh = agg_results['agg_thresh']

    plot_loss_vs_epoch(
        agg_epoch,
        title=f'Training Loss vs Epoch — {model.upper()} ({exp_name})',
        save_path=plots_dir / 'loss_vs_epoch',
    )
    plot_loss_vs_step(
        agg_step,
        title=f'Training Loss vs Step — {model.upper()} ({exp_name})',
        save_path=plots_dir / 'loss_vs_step',
    )
    plot_early_stage_zoom(
        agg_epoch,
        title=f'Early-Stage Zoom (Epochs 1–5) — {model.upper()} ({exp_name})',
        save_path=plots_dir / 'early_stage_zoom',
        zoom_epochs=5,
    )
    plot_threshold_comparison(
        agg_thresh,
        title=f'Steps to Reach Loss Thresholds — {model.upper()} ({exp_name})',
        save_path=plots_dir / 'threshold_comparison',
    )
    print(f"Plots saved to {plots_dir}/")
