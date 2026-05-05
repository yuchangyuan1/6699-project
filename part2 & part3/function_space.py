"""
§10 Function-space similarity.

For each matched seed, evaluate the SGD and Adam models on the test set and
compare them as functions:

    D_pred  = (1/N) Σ 1[ argmax p_S(x) ≠ argmax p_A(x) ]
    D_SKL   = (1/2N) Σ ( KL(p_S || p_A) + KL(p_A || p_S) )
    C_logit = (1/N) Σ <z_S(x), z_A(x)> / (‖z_S(x)‖ · ‖z_A(x)‖)

These metrics target three different aspects of function similarity:
    * D_pred  — discrete prediction agreement (decision-level)
    * D_SKL   — full softmax distribution divergence
    * C_logit — pre-softmax direction similarity (scale-invariant)

Run:
    python function_space.py
Output:
    results_part2/function_space.json
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_io import load_final_model
from data_utils import get_loaders


SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]
RESULTS_DIR = "./results_part2"
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                 "function_space.json"))
    return p.parse_args()


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits_all.append(model(xb).cpu())
        labels_all.append(yb.cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def function_space_metrics(z_s: torch.Tensor, z_a: torch.Tensor):
    """All three function-space comparisons. Inputs are (N, C) logits."""
    # Predictions
    pred_s = z_s.argmax(1)
    pred_a = z_a.argmax(1)
    d_pred = (pred_s != pred_a).float().mean().item()

    # Symmetric KL via softmax
    p_s = F.softmax(z_s, dim=1)
    p_a = F.softmax(z_a, dim=1)
    log_s = F.log_softmax(z_s, dim=1)
    log_a = F.log_softmax(z_a, dim=1)
    kl_sa = (p_s * (log_s - log_a)).sum(1)
    kl_as = (p_a * (log_a - log_s)).sum(1)
    d_skl = 0.5 * (kl_sa + kl_as).mean().item()

    # Logit cosine similarity
    z_s_n = z_s.norm(dim=1, keepdim=True).clamp_min(EPS)
    z_a_n = z_a.norm(dim=1, keepdim=True).clamp_min(EPS)
    c_logit = ((z_s * z_a).sum(1) / (z_s_n.squeeze(1) * z_a_n.squeeze(1))).mean().item()

    return {"D_pred": d_pred, "D_SKL": d_skl, "C_logit": c_logit}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out = {"runs": []}
    t0 = time.time()

    for arch in ARCHS:
        for seed in SEEDS:
            try:
                m_sgd = load_final_model(arch, "SGD", seed, device)
                m_ada = load_final_model(arch, "Adam", seed, device)
            except FileNotFoundError as e:
                print(f"SKIP {arch}/seed={seed}: {e}", flush=True)
                continue

            _, test_loader = get_loaders(seed)
            z_s, y_s = collect_logits(m_sgd, test_loader, device)
            z_a, y_a = collect_logits(m_ada, test_loader, device)
            assert torch.equal(y_s, y_a), "test loaders must produce same order"

            metrics = function_space_metrics(z_s, z_a)
            metrics.update({"arch": arch, "seed": seed})
            out["runs"].append(metrics)
            print(f"  {arch}/seed={seed}: "
                  f"D_pred={metrics['D_pred']:.4f}  "
                  f"D_SKL={metrics['D_SKL']:.4f}  "
                  f"C_logit={metrics['C_logit']:.4f}", flush=True)

    # Aggregate per architecture
    agg = {}
    for r in out["runs"]:
        d = agg.setdefault(r["arch"], {"D_pred": [], "D_SKL": [], "C_logit": []})
        d["D_pred"].append(r["D_pred"])
        d["D_SKL"].append(r["D_SKL"])
        d["C_logit"].append(r["C_logit"])
    out["aggregated"] = {
        a: {
            "D_pred_mean":  float(np.mean(v["D_pred"])),
            "D_pred_std":   float(np.std(v["D_pred"], ddof=0)),
            "D_SKL_mean":   float(np.mean(v["D_SKL"])),
            "D_SKL_std":    float(np.std(v["D_SKL"], ddof=0)),
            "C_logit_mean": float(np.mean(v["C_logit"])),
            "C_logit_std":  float(np.std(v["C_logit"], ddof=0)),
            "n_seeds":      len(v["D_pred"]),
        } for a, v in agg.items()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
