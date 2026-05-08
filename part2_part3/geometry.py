"""
Hessian-based geometry metrics:

  hessian_vector_product   - Hv via second-order autograd (Pearlmutter trick)
  sharpness_power_iter     - dominant eigenvalue lambda_max(H) via power iteration
  hessian_trace_hutchinson - tr(H) via Hutchinson's randomized estimator
  param_distance           - ||theta - theta_0||_2
  pairwise_mean_distance   - mean pairwise L2 between flat parameter vectors
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn


def hessian_vector_product(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    v_flat: torch.Tensor,
) -> torch.Tensor:
    """Hv where H = d^2 L / d theta^2, computed via Hv = grad((grad L)^T v)."""
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])
    grad_v = (grad_flat * v_flat.detach()).sum()
    hv_list = torch.autograd.grad(grad_v, params, retain_graph=False)
    return torch.cat([h.detach().flatten() for h in hv_list])


def sharpness_power_iter(
    model: nn.Module,
    loader,
    device: torch.device,
    n_steps: int = 50,
    tol: float = 1e-3,
    max_samples: int = 512,
) -> float:
    """Estimate lambda_max(H) by power iteration on a fixed mini-batch."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]

    x_list, y_list, total = [], [], 0
    for xb, yb in loader:
        x_list.append(xb); y_list.append(yb)
        total += xb.size(0)
        if total >= max_samples:
            break
    x_batch = torch.cat(x_list)[:max_samples].to(device)
    y_batch = torch.cat(y_list)[:max_samples].to(device)

    d = sum(p.numel() for p in params)
    v = torch.randn(d, device=device)
    v = v / v.norm()

    eigenvalue = 0.0
    for _ in range(n_steps):
        model.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        hv = hessian_vector_product(loss, params, v)
        new_eigenvalue = (v * hv).sum().item()
        v = hv / (hv.norm() + 1e-12)
        if abs(new_eigenvalue - eigenvalue) < tol:
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue
    return float(eigenvalue)


def hessian_trace_hutchinson(
    model: nn.Module,
    loader,
    device: torch.device,
    n_samples: int = 30,
    max_samples: int = 512,
) -> float:
    """Estimate tr(H) by averaging z^T H z over Rademacher z."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)

    x_list, y_list, total = [], [], 0
    for xb, yb in loader:
        x_list.append(xb); y_list.append(yb)
        total += xb.size(0)
        if total >= max_samples:
            break
    x_batch = torch.cat(x_list)[:max_samples].to(device)
    y_batch = torch.cat(y_list)[:max_samples].to(device)

    estimates = []
    for _ in range(n_samples):
        z = torch.randint(0, 2, (d,), device=device).float() * 2 - 1
        model.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        hz = hessian_vector_product(loss, params, z)
        estimates.append((z * hz).sum().item())
    return float(np.mean(estimates))


def param_distance(flat_current: torch.Tensor, flat_init: torch.Tensor) -> float:
    return (flat_current - flat_init).norm().item()


def pairwise_mean_distance(flat_params_list: List[torch.Tensor]) -> float:
    """Mean pairwise L2 distance over all C(n,2) unordered pairs."""
    n = len(flat_params_list)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (flat_params_list[i] - flat_params_list[j]).norm().item()
            count += 1
    return total / count
