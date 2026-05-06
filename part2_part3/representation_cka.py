"""
§11 Representation-space geometry.

For each matched seed, extract penultimate-layer features for the same set of
test samples from the SGD and Adam models, then compare:

    1. Feature-kernel alignment (cosine Gram matrices, no centering):

           A(K_S, K_A) = <K_S, K_A>_F / (||K_S||_F · ||K_A||_F)

    2. Linear CKA (centered):

           CKA = ||F̃_S^T F̃_A||_F^2 / (||F̃_S^T F̃_S||_F · ||F̃_A^T F̃_A||_F)

       where F̃ is the column-centered feature matrix.

Penultimate layer = ReLU output before the final Linear(128, 10) classifier:
    MLP:      net[5]
    SmallCNN: classifier[1]

Run:
    python representation_cka.py
Output:
    results_part2/representation_cka.json
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


SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]
RESULTS_DIR = "./results_part2"
N_SAMPLES = 1024  # enough for stable CKA, fits comfortably on GPU
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_SAMPLES)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                 "representation_cka.json"))
    return p.parse_args()


def get_penultimate_module(model: nn.Module) -> nn.Module:
    """Return the module whose output is the penultimate-layer feature."""
    if hasattr(model, "net"):       # MLP
        return model.net[5]
    if hasattr(model, "classifier"):  # SmallCNN
        return model.classifier[1]
    raise ValueError(f"Unknown architecture: {type(model)}")


@torch.no_grad()
def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward `x` through `model` and capture penultimate-layer activations."""
    features = []
    handle = get_penultimate_module(model).register_forward_hook(
        lambda mod, inp, out: features.append(out.detach())
    )
    try:
        model.eval()
        _ = model(x)
    finally:
        handle.remove()
    return features[0]


def feature_kernel_alignment(F_s: torch.Tensor, F_a: torch.Tensor) -> float:
    """
    Cosine-normalized Gram alignment:
        K(i,j) = <h_i, h_j> / (‖h_i‖ ‖h_j‖)
        A = <K_S, K_A>_F / (‖K_S‖_F · ‖K_A‖_F)
    """
    def _gram(F):
        norms = F.norm(dim=1, keepdim=True).clamp_min(EPS)
        Fn = F / norms
        return Fn @ Fn.T
    K_s = _gram(F_s)
    K_a = _gram(F_a)
    num = (K_s * K_a).sum().item()
    den = (K_s.norm().item() * K_a.norm().item() + EPS)
    return num / den


def linear_cka(F_s: torch.Tensor, F_a: torch.Tensor) -> float:
    """Centered linear CKA, computed in feature space (efficient when d << n)."""
    F_s = F_s - F_s.mean(dim=0, keepdim=True)
    F_a = F_a - F_a.mean(dim=0, keepdim=True)
    num = (F_s.T @ F_a).norm().pow(2).item()
    den = ((F_s.T @ F_s).norm().item()
           * (F_a.T @ F_a).norm().item()
           + EPS)
    return num / den


@torch.no_grad()
def collect_test_inputs(loader, n_max: int, device):
    xs, ys, total = [], [], 0
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
        total += xb.size(0)
        if total >= n_max:
            break
    return torch.cat(xs)[:n_max].to(device), torch.cat(ys)[:n_max].to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out = {"config": {"n_samples": args.n}, "runs": []}
    t0 = time.time()

    # Sanity check 1: CKA(K, K) = 1 on a synthetic feature
    rand = torch.randn(64, 32, device=device)
    assert abs(linear_cka(rand, rand) - 1.0) < 1e-5, "linear_cka self-similarity broken"
    assert abs(feature_kernel_alignment(rand, rand) - 1.0) < 1e-5, \
        "feature_kernel_alignment self-similarity broken"
    # Sanity check 2: CKA invariant to isotropic scaling
    assert abs(linear_cka(rand, 5.0 * rand) - 1.0) < 1e-5, "linear_cka scale invariance broken"

    for arch in ARCHS:
        for seed in SEEDS:
            try:
                m_sgd = load_final_model(arch, "SGD", seed, device)
                m_ada = load_final_model(arch, "Adam", seed, device)
            except FileNotFoundError as e:
                print(f"SKIP {arch}/seed={seed}: {e}", flush=True)
                continue

            _, test_loader = get_loaders(seed)
            x, _ = collect_test_inputs(test_loader, args.n, device)

            F_s = extract_features(m_sgd, x)
            F_a = extract_features(m_ada, x)

            A   = feature_kernel_alignment(F_s, F_a)
            cka = linear_cka(F_s, F_a)
            entry = {
                "arch": arch, "seed": seed,
                "feature_kernel_alignment": A,
                "linear_cka":               cka,
                "n_samples":                F_s.shape[0],
                "feature_dim":              F_s.shape[1],
            }
            out["runs"].append(entry)
            print(f"  {arch}/seed={seed}: "
                  f"A={A:.4f}  CKA={cka:.4f}  d={F_s.shape[1]}", flush=True)

    agg = {}
    for r in out["runs"]:
        d = agg.setdefault(r["arch"], {"A": [], "CKA": []})
        d["A"].append(r["feature_kernel_alignment"])
        d["CKA"].append(r["linear_cka"])
    out["aggregated"] = {
        a: {
            "A_mean":   float(np.mean(v["A"])),
            "A_std":    float(np.std(v["A"], ddof=0)),
            "CKA_mean": float(np.mean(v["CKA"])),
            "CKA_std":  float(np.std(v["CKA"], ddof=0)),
            "n_seeds":  len(v["A"]),
        } for a, v in agg.items()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
