"""Aggregate per-seed CSV results and write combined summaries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_run_results(base_dir: Path, optimizer_name: str, seeds: list) -> dict:
    epoch_dfs, step_dfs, threshold_dfs = [], [], []
    for seed in seeds:
        d = base_dir / optimizer_name / f"seed_{seed}"
        epoch_dfs.append(pd.read_csv(d / 'metrics.csv'))
        step_dfs.append(pd.read_csv(d / 'step_metrics.csv'))
        threshold_dfs.append(pd.read_csv(d / 'thresholds.csv'))
    return {'epoch_dfs': epoch_dfs, 'step_dfs': step_dfs, 'threshold_dfs': threshold_dfs}


def aggregate_epoch_metrics(epoch_dfs, optimizer_name):
    combined = pd.concat([df.assign(seed=i) for i, df in enumerate(epoch_dfs)])
    agg = combined.groupby('epoch')['loss'].agg(['mean', 'std']).reset_index()
    agg.columns = ['epoch', 'mean_loss', 'std_loss']
    agg['std_loss'] = agg['std_loss'].fillna(0.0)
    agg['optimizer'] = optimizer_name
    return agg


def aggregate_step_metrics(step_dfs, optimizer_name, smooth_window=50):
    combined = pd.concat([df.assign(seed_idx=i) for i, df in enumerate(step_dfs)])
    agg = combined.groupby('global_step')['loss'].agg(['mean', 'std']).reset_index()
    agg.columns = ['global_step', 'mean_loss', 'std_loss']
    agg['std_loss'] = agg['std_loss'].fillna(0.0)
    agg['mean_loss_smooth'] = agg['mean_loss'].rolling(smooth_window, min_periods=1, center=True).mean()
    agg['optimizer'] = optimizer_name
    return agg


def aggregate_threshold_stats(threshold_dfs, optimizer_name):
    combined = pd.concat(threshold_dfs)
    rows = []
    for threshold, group in combined.groupby('threshold'):
        valid_epoch = group['first_epoch'].dropna()
        valid_step  = group['first_global_step'].dropna()
        rows.append({
            'optimizer': optimizer_name,
            'threshold': threshold,
            'n_seeds': len(group),
            'n_reached': len(valid_epoch),
            'mean_first_epoch': valid_epoch.mean() if len(valid_epoch) else None,
            'std_first_epoch':  valid_epoch.std()  if len(valid_epoch) > 1 else 0.0,
            'mean_first_global_step': valid_step.mean() if len(valid_step) else None,
            'std_first_global_step':  valid_step.std()  if len(valid_step) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def build_aggregated_outputs(cfg, base_dir: Path) -> dict:
    seeds = cfg['experiment']['seeds']
    optimizers = list(cfg['optimizers'].keys())

    agg_epoch, agg_step, agg_thresh = {}, {}, {}
    epoch_parts, step_parts, thresh_parts = [], [], []
    for opt in optimizers:
        results = load_run_results(base_dir, opt, seeds)
        agg_epoch[opt]  = aggregate_epoch_metrics(results['epoch_dfs'], opt)
        agg_step[opt]   = aggregate_step_metrics(results['step_dfs'], opt)
        agg_thresh[opt] = aggregate_threshold_stats(results['threshold_dfs'], opt)
        epoch_parts.append(agg_epoch[opt])
        step_parts.append(agg_step[opt])
        thresh_parts.append(agg_thresh[opt])

    pd.concat(epoch_parts,  ignore_index=True).to_csv(base_dir / 'aggregated_metrics.csv', index=False)
    pd.concat(step_parts,   ignore_index=True).to_csv(base_dir / 'aggregated_step_metrics.csv', index=False)
    pd.concat(thresh_parts, ignore_index=True).to_csv(base_dir / 'aggregated_thresholds.csv', index=False)

    summary_md = generate_report_summary(cfg, agg_epoch, agg_thresh)
    (base_dir / 'report_ready_summary.md').write_text(summary_md, encoding='utf-8')

    return {'agg_epoch': agg_epoch, 'agg_step': agg_step, 'agg_thresh': agg_thresh}


def generate_report_summary(cfg, agg_epoch, agg_thresh) -> str:
    exp_name = cfg['experiment']['name']
    model    = cfg['experiment']['model']
    seeds    = cfg['experiment']['seeds']
    thresholds = cfg['metrics']['loss_thresholds']
    epochs   = cfg['training']['epochs']

    lines = [
        f"# Experiment Summary: {exp_name}", "",
        f"**Model**: {model}  ", f"**Dataset**: Fashion-MNIST  ",
        f"**Seeds**: {seeds}  ", f"**Epochs**: {epochs}  ", "",
        "## Final Epoch Loss (mean +/- std)", "",
        "| Optimizer | Final Loss (mean) | Final Loss (std) |",
        "|-----------|-------------------|------------------|",
    ]
    for opt, df in agg_epoch.items():
        last = df[df['epoch'] == df['epoch'].max()].iloc[0]
        lines.append(f"| {opt.upper()} | {last['mean_loss']:.4f} | {last['std_loss']:.4f} |")

    lines += [
        "", "## Steps to Reach Loss Thresholds (mean +/- std)", "",
        "| Threshold | SGD (mean steps) | SGD (std) | Adam (mean steps) | Adam (std) | Faster |",
        "|-----------|------------------|-----------|-------------------|------------|--------|",
    ]
    for t in sorted(thresholds, reverse=True):
        sgd_mean = sgd_std = adam_mean = adam_std = None
        for opt, df in agg_thresh.items():
            row = df[df['threshold'] == t]
            if len(row) == 0: continue
            r = row.iloc[0]
            if opt == 'sgd':  sgd_mean, sgd_std  = r['mean_first_global_step'], r['std_first_global_step']
            else:             adam_mean, adam_std = r['mean_first_global_step'], r['std_first_global_step']

        def fmt(m, s):
            if m is None or (isinstance(m, float) and np.isnan(m)):
                return "N/A | N/A"
            return f"{m:.0f} | {s:.0f}"

        faster = ""
        if sgd_mean and adam_mean and not np.isnan(sgd_mean) and not np.isnan(adam_mean):
            faster = "Adam" if adam_mean < sgd_mean else "SGD"

        lines.append(f"| {t} | {fmt(sgd_mean, sgd_std)} | {fmt(adam_mean, adam_std)} | {faster} |")

    return "\n".join(lines) + "\n"
