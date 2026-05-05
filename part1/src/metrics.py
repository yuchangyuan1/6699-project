"""Metric data structures, threshold statistics, gradient norm."""
from typing import NamedTuple, Optional
import pandas as pd
import torch
import torch.nn as nn


class EpochRecord(NamedTuple):
    epoch: int
    loss: float


class StepRecord(NamedTuple):
    epoch: int
    step: int
    global_step: int
    loss: float


def compute_threshold_stats(
    step_records: list,
    thresholds: list,
) -> dict:
    """
    For each threshold, find the first StepRecord where loss <= threshold.
    Returns {threshold: {"epoch": int, "step": int, "global_step": int} | None}.
    """
    # Sort thresholds descending so we can mark once we pass each
    remaining = {t: True for t in thresholds}
    result = {t: None for t in thresholds}

    for rec in step_records:
        for t in list(remaining.keys()):
            if rec.loss <= t:
                result[t] = {
                    'epoch': rec.epoch,
                    'step': rec.step,
                    'global_step': rec.global_step,
                }
                del remaining[t]
        if not remaining:
            break

    return result


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm without clipping. Call after loss.backward()."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


def records_to_epoch_df(epoch_records: list) -> pd.DataFrame:
    return pd.DataFrame(epoch_records, columns=['epoch', 'loss'])


def records_to_step_df(step_records: list) -> pd.DataFrame:
    return pd.DataFrame(step_records, columns=['epoch', 'step', 'global_step', 'loss'])


def thresholds_to_df(threshold_stats: dict, optimizer_name: str, seed: int) -> pd.DataFrame:
    rows = []
    for t, info in threshold_stats.items():
        row = {'optimizer': optimizer_name, 'seed': seed, 'threshold': t}
        if info is not None:
            row['first_epoch'] = info['epoch']
            row['first_step'] = info['step']
            row['first_global_step'] = info['global_step']
        else:
            row['first_epoch'] = None
            row['first_step'] = None
            row['first_global_step'] = None
        rows.append(row)
    return pd.DataFrame(rows)
