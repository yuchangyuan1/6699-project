"""Sweep learning rates for SGD and Adam on the MLP.

Usage:
    python scripts/run_lr_search.py --config configs/experiment_c_lr_search.yaml
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import torch.nn as nn

from src.datasets import get_fashion_mnist_loaders
from src.models import build_model, build_optimizer
from src.metrics import StepRecord, compute_threshold_stats
from src.utils import get_device, load_config, reset_parameters, set_seed


def run_lr_single(cfg, optimizer_name, lr, seed, output_dir):
    set_seed(seed)
    device = get_device()
    model = build_model(cfg['experiment']['model'])
    reset_parameters(model)
    model.to(device)

    lr_cfg = dict(cfg['lr_search'][optimizer_name])
    lr_cfg['lr'] = lr
    lr_cfg.pop('lr_candidates', None)
    optimizer = build_optimizer(optimizer_name, model, lr_cfg)

    data_cfg = cfg['data']
    train_loader, _ = get_fashion_mnist_loaders(
        data_dir=data_cfg['data_dir'],
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        seed=seed,
    )

    epochs = cfg['training']['epochs']
    criterion = nn.CrossEntropyLoss()
    model.train()

    epoch_records = []
    step_records = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss_sum, epoch_steps = 0.0, 0
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_val = loss.item()
            step_records.append((epoch, step, global_step, loss_val))
            epoch_loss_sum += loss_val
            epoch_steps += 1
        epoch_records.append({'epoch': epoch, 'loss': epoch_loss_sum / epoch_steps})

    thresholds = cfg['metrics']['loss_thresholds']
    step_recs = [StepRecord(*r) for r in step_records]
    thr_stats = compute_threshold_stats(step_recs, thresholds)
    final_loss = epoch_records[-1]['loss']
    steps_to_10 = thr_stats.get(1.0, {}) or {}
    return pd.DataFrame(epoch_records), final_loss, steps_to_10.get('global_step', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = Path(cfg['output']['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg['experiment']['seeds'][0]

    summary_rows = []
    for opt_name, opt_cfg in cfg['lr_search'].items():
        print(f"\n=== {opt_name.upper()} LR Search ===")
        for lr in opt_cfg['lr_candidates']:
            print(f"  lr={lr:.0e} ...", end=' ', flush=True)
            run_dir = base_dir / opt_name / f"lr_{lr:.0e}"
            _, final_loss, steps_to_10 = run_lr_single(cfg, opt_name, lr, seed, run_dir)
            print(f"final_loss={final_loss:.4f}")
            summary_rows.append({
                'optimizer': opt_name,
                'lr': lr,
                'final_epoch_loss': final_loss,
                'steps_to_loss_1.0': steps_to_10,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / 'lr_search_summary.csv', index=False)
    print(f"\nLR search summary saved to {base_dir / 'lr_search_summary.csv'}")

    print("\n=== Best LRs (lowest final loss) ===")
    for opt_name in cfg['lr_search']:
        sub = summary_df[summary_df['optimizer'] == opt_name]
        best = sub.loc[sub['final_epoch_loss'].idxmin()]
        print(f"  {opt_name.upper()}: best lr={best['lr']:.0e}, final_loss={best['final_epoch_loss']:.4f}")


if __name__ == '__main__':
    main()
