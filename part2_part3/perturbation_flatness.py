"""
Perturbation flatness: for each trained model, evaluate

    Delta L_hat(sigma) = L_hat(theta + delta) - L_hat(theta)

with delta a relative-global Gaussian perturbation
    delta = sigma * ||theta||_2 / ||xi||_2 * xi,   xi ~ N(0, I).

Operational check on the Hessian-trace flatness proxy.

Output: results_part2/perturbation_flatness.json
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

from model_io import load_final_model
from data_utils import get_loaders


SIGMAS       = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
K_PERT       = 10
EVAL_SAMPLES = 4096
RESULTS_DIR  = "./results_part2"
SEEDS        = [42, 123, 456, 789, 2024]
ARCH_OPTS    = [("MLP", "SGD"), ("MLP", "Adam"),
                ("SmallCNN", "SGD"), ("SmallCNN", "Adam")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=K_PERT)
    p.add_argument("--samples", type=int, default=EVAL_SAMPLES)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "perturbation_flatness.json"))
    return p.parse_args()


@torch.no_grad()
def collect_eval_batch(loader, n_max, device):
    xs, ys, total = [], [], 0
    for xb, yb in loader:
        xs.append(xb); ys.append(yb)
        total += xb.size(0)
        if total >= n_max:
            break
    return torch.cat(xs)[:n_max].to(device), torch.cat(ys)[:n_max].to(device)


@torch.no_grad()
def eval_loss(model, x, y) -> float:
    model.eval()
    return float(nn.CrossEntropyLoss()(model(x), y).item())


def get_param_list(model):
    return [p for p in model.parameters() if p.requires_grad]


def add_perturbation_(params, delta_chunks):
    with torch.no_grad():
        for p, d in zip(params, delta_chunks):
            p.add_(d)


def sub_perturbation_(params, delta_chunks):
    with torch.no_grad():
        for p, d in zip(params, delta_chunks):
            p.sub_(d)


def make_global_perturbation(params, sigma, device):
    """Return per-tensor perturbations whose joint norm equals sigma * ||theta||_2."""
    theta_norm = (sum((p ** 2).sum().item() for p in params)) ** 0.5
    xi_chunks  = [torch.randn_like(p) for p in params]
    xi_norm    = (sum((x ** 2).sum().item() for x in xi_chunks)) ** 0.5
    scale = sigma * theta_norm / max(xi_norm, 1e-12)
    return [scale * x for x in xi_chunks]


def perturbation_curve(model, x, y, sigmas, K, device):
    base_loss = eval_loss(model, x, y)
    params = get_param_list(model)

    out = {"base_loss": base_loss, "by_sigma": {}}
    for sigma in sigmas:
        deltas = []
        for _ in range(K):
            d_chunks = make_global_perturbation(params, sigma, device)
            add_perturbation_(params, d_chunks)
            new_loss = eval_loss(model, x, y)
            sub_perturbation_(params, d_chunks)
            deltas.append(new_loss - base_loss)
        deltas = np.array(deltas)
        out["by_sigma"][f"{sigma:.0e}"] = {
            "sigma": sigma,
            "mean":  float(deltas.mean()),
            "std":   float(deltas.std(ddof=0)),
            "values": deltas.tolist(),
        }
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, _ = get_loaders(seed=42)
    x, y = collect_eval_batch(train_loader, args.samples, device)
    print(f"Eval batch: {x.shape}", flush=True)

    results = {
        "config": {"sigmas": SIGMAS, "K": args.K, "eval_samples": args.samples},
        "runs":   [],
    }

    t0 = time.time()
    for arch, opt in ARCH_OPTS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                model = load_final_model(arch, opt, seed, device)
            except FileNotFoundError as e:
                print(f"SKIP {arch}/{opt}/seed={seed}: {e}", flush=True)
                continue

            t1 = time.time()
            curve = perturbation_curve(model, x, y, SIGMAS, args.K, device)
            results["runs"].append({
                "arch": arch, "optimizer": opt, "seed": seed,
                "base_loss": curve["base_loss"],
                "by_sigma":  curve["by_sigma"],
            })
            print(f"  {arch}/{opt}/seed={seed}: base_loss={curve['base_loss']:.4f}  "
                  f"DeltaL(1e-3)={curve['by_sigma']['1e-03']['mean']:.4e}  "
                  f"({time.time()-t1:.1f}s)", flush=True)

    agg = {}
    for r in results["runs"]:
        key = f"{r['arch']}_{r['optimizer']}"
        agg.setdefault(key, {"sigma_means": {}, "n_seeds": 0})
        agg[key]["n_seeds"] += 1
        for sk, sv in r["by_sigma"].items():
            agg[key]["sigma_means"].setdefault(sk, []).append(sv["mean"])
    aggregated = {
        key: {
            "n_seeds": val["n_seeds"],
            "sigmas": {sk: {"mean": float(np.mean(vs)),
                            "std":  float(np.std(vs, ddof=0))}
                       for sk, vs in val["sigma_means"].items()},
        } for key, val in agg.items()
    }
    results["aggregated"] = aggregated

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
