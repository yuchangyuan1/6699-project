"""Utility functions: seeding, config, I/O."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_parameters(model: nn.Module) -> None:
    """Re-initialize all weights deterministically (call after set_seed)."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_output_dir(base_dir: str, optimizer_name: str, seed: int) -> Path:
    d = Path(base_dir) / optimizer_name / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
