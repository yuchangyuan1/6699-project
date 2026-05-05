"""Core training loop and single-run experiment orchestration."""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets import get_fashion_mnist_loaders
from src.metrics import (
    EpochRecord, StepRecord,
    compute_gradient_norm, compute_threshold_stats,
    records_to_epoch_df, records_to_step_df, thresholds_to_df,
)
from src.models import build_model, build_optimizer
from src.utils import get_device, get_output_dir, reset_parameters, set_seed


def train_one_run(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    epochs: int,
    device: torch.device,
    log_grad_norm: bool = False,
) -> tuple:
    """
    Run training and return (epoch_records, step_records, grad_norm_records).
    grad_norm_records is empty if log_grad_norm=False.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    epoch_records = []
    step_records = []
    grad_norm_records = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if log_grad_norm:
                gn = compute_gradient_norm(model)
                grad_norm_records.append((epoch, step, global_step + 1, gn))

            optimizer.step()
            global_step += 1
            loss_val = loss.item()
            step_records.append(StepRecord(epoch, step, global_step, loss_val))
            epoch_loss_sum += loss_val
            epoch_steps += 1
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

        epoch_loss = epoch_loss_sum / epoch_steps
        epoch_records.append(EpochRecord(epoch, epoch_loss))

    return epoch_records, step_records, grad_norm_records


def run_experiment(
    cfg: dict,
    optimizer_name: str,
    seed: int,
    output_dir: Path,
) -> dict:
    """
    Full single-run experiment: seed → model → optimizer → train → save.
    Returns a summary dict.
    """
    set_seed(seed)

    device = get_device()
    model_name = cfg['experiment']['model']
    model = build_model(model_name)
    reset_parameters(model)  # deterministic init after seeding

    opt_cfg = cfg['optimizers'][optimizer_name]
    optimizer = build_optimizer(optimizer_name, model, opt_cfg)

    data_cfg = cfg['data']
    train_loader, _ = get_fashion_mnist_loaders(
        data_dir=data_cfg['data_dir'],
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        seed=seed,
    )

    epochs = cfg['training']['epochs']
    log_grad_norm = cfg['metrics'].get('log_grad_norm', False)
    thresholds = cfg['metrics']['loss_thresholds']

    t0 = time.time()
    epoch_records, step_records, grad_norm_records = train_one_run(
        model, optimizer, train_loader, epochs, device, log_grad_norm
    )
    elapsed = time.time() - t0

    threshold_stats = compute_threshold_stats(step_records, thresholds)

    # --- Save metrics ---
    epoch_df = records_to_epoch_df(epoch_records)
    step_df = records_to_step_df(step_records)
    thr_df = thresholds_to_df(threshold_stats, optimizer_name, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_df.to_csv(output_dir / 'metrics.csv', index=False)
    step_df.to_csv(output_dir / 'step_metrics.csv', index=False)
    thr_df.to_csv(output_dir / 'thresholds.csv', index=False)

    if grad_norm_records:
        import pandas as pd
        gn_df = pd.DataFrame(grad_norm_records, columns=['epoch', 'step', 'global_step', 'grad_norm'])
        gn_df.to_csv(output_dir / 'grad_norm.csv', index=False)

    # --- Summary ---
    summary = {
        'experiment': cfg['experiment']['name'],
        'model': model_name,
        'optimizer': optimizer_name,
        'optimizer_cfg': opt_cfg,
        'seed': seed,
        'epochs': epochs,
        'batch_size': data_cfg['batch_size'],
        'final_epoch_loss': epoch_records[-1].loss,
        'training_time_sec': round(elapsed, 2),
        'device': str(device),
        'thresholds': {
            str(k): v for k, v in threshold_stats.items()
        },
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
