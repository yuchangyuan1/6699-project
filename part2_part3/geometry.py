"""
Part II geometry metrics for solution analysis.

Implements:
  - hessian_vector_product  : Hv via second-order autograd (Pearlmutter trick)
  - sharpness_power_iter    : dominant eigenvalue lambda_max(H) via power iteration
  - hessian_trace_hutchinson: tr(H) via Hutchinson's randomized estimator
  - gradient_norm           : mean ||g||_2 over a mini-batch
  - param_distance          : ||theta - theta_0||_2
  - pairwise_distances      : mean pairwise L2 between a list of flat param vectors
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Hessian-vector product (Pearlmutter trick, works on CPU and GPU)
# ---------------------------------------------------------------------------

def hessian_vector_product(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    v_flat: torch.Tensor,
) -> torch.Tensor:
    """
    Compute H*v where H = d^2 L / d theta^2.

    Uses the identity:  Hv = grad( (grad L)^T v )

    Args:
        loss   : scalar loss (already computed, with create_graph=True)
        params : list of parameter tensors (same order as v_flat is built from)
        v_flat : flat 1-D direction vector, same length as total parameters

    Returns:
        Hv as a flat 1-D tensor (detached, no grad)
    """
    # First-order gradient, keeping the graph so we can differentiate again
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])

    # Split v_flat back into per-parameter chunks
    grad_v = (grad_flat * v_flat.detach()).sum()

    # Second derivative: d/dtheta [ grad_flat . v ]
    hv_list = torch.autograd.grad(grad_v, params, retain_graph=False)
    hv_flat = torch.cat([h.detach().flatten() for h in hv_list])
    return hv_flat


# ---------------------------------------------------------------------------
# Dominant eigenvalue via power iteration
# ---------------------------------------------------------------------------

def sharpness_power_iter(
    model: nn.Module,
    loader,
    device: torch.device,
    n_steps: int = 50,
    tol: float = 1e-3,
    max_samples: int = 512,
) -> float:
    """
    Estimate lambda_max(H) by power iteration on a mini-batch.

    The Hessian is approximated on a single mini-batch of size <= max_samples.
    Returns the estimated dominant eigenvalue (a float).

    Mathematical background:
        Starting from a random unit vector v_0, iterate:
            v_{k+1} = H v_k / ||H v_k||_2
        The Rayleigh quotient v_k^T H v_k converges to lambda_max.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]

    # Collect one mini-batch (at most max_samples examples)
    x_list, y_list = [], []
    total = 0
    for xb, yb in loader:
        x_list.append(xb)
        y_list.append(yb)
        total += xb.size(0)
        if total >= max_samples:
            break
    x_batch = torch.cat(x_list)[:max_samples].to(device)
    y_batch = torch.cat(y_list)[:max_samples].to(device)

    # Random unit vector
    d = sum(p.numel() for p in params)
    v = torch.randn(d, device=device)
    v = v / v.norm()

    eigenvalue = 0.0
    for _ in range(n_steps):
        # Recompute loss with graph each iteration
        model.zero_grad()
        loss = criterion(model(x_batch), y_batch)

        hv = hessian_vector_product(loss, params, v)

        # Rayleigh quotient estimate
        new_eigenvalue = (v * hv).sum().item()

        # Update direction
        v = hv / (hv.norm() + 1e-12)

        if abs(new_eigenvalue - eigenvalue) < tol:
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue

    return float(eigenvalue)


# ---------------------------------------------------------------------------
# Hessian trace via Hutchinson's estimator
# ---------------------------------------------------------------------------

def hessian_trace_hutchinson(
    model: nn.Module,
    loader,
    device: torch.device,
    n_samples: int = 30,
    max_samples: int = 512,
) -> float:
    """
    Estimate tr(H) using Hutchinson's randomized trace estimator.

    For random vectors z ~ N(0, I):
        tr(H) = E[ z^T H z ]

    We average over n_samples independent draws of z.

    Args:
        model      : trained model in eval mode
        loader     : data loader (used for one mini-batch)
        device     : torch device
        n_samples  : number of Hutchinson probe vectors
        max_samples: mini-batch size cap for Hessian computation

    Returns:
        Estimated Hessian trace (float)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)

    # Collect mini-batch
    x_list, y_list = [], []
    total = 0
    for xb, yb in loader:
        x_list.append(xb)
        y_list.append(yb)
        total += xb.size(0)
        if total >= max_samples:
            break
    x_batch = torch.cat(x_list)[:max_samples].to(device)
    y_batch = torch.cat(y_list)[:max_samples].to(device)

    trace_estimates = []
    for _ in range(n_samples):
        # Rademacher vector: z_i in {-1, +1} uniformly
        z = torch.randint(0, 2, (d,), device=device).float() * 2 - 1

        model.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        hz = hessian_vector_product(loss, params, z)

        # z^T H z
        trace_estimates.append((z * hz).sum().item())

    return float(np.mean(trace_estimates))


# ---------------------------------------------------------------------------
# Utility: parameter distance and inter-seed dispersion
# ---------------------------------------------------------------------------

def param_distance(flat_current: torch.Tensor, flat_init: torch.Tensor) -> float:
    """||theta_t - theta_0||_2"""
    return (flat_current - flat_init).norm().item()


def pairwise_mean_distance(flat_params_list: List[torch.Tensor]) -> float:
    """
    Mean pairwise L2 distance between all pairs in flat_params_list.

    For n vectors, computes C(n,2) = n*(n-1)/2 pairwise distances
    and returns the mean.
    """
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
