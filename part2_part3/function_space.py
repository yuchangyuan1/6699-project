"""
Function-space similarity between matched-seed SGD and Adam models on test:

    D_pred  - prediction-disagreement rate (decision level)
    D_SKL   - mean symmetric KL of softmax distributions
    C_logit - mean cosine similarity of pre-softmax logits

Output: results_part2/function_space.json
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
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "function_space.json"))
    return p.parse_args()


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for xb, yb in loader:
        logits.append(model(xb.to(device)).cpu())
        labels.append(yb)
    return torch.cat(logits), torch.cat(labels)


def function_space_metrics(z_s, z_a):
    """Three function-space comparisons given (N, C) logits z_s, z_a."""
    pred_s = z_s.argmax(1); pred_a = z_a.argmax(1)
    d_pred = (pred_s != pred_a).float().mean().item()

    p_s = F.softmax(z_s, dim=1);   p_a = F.softmax(z_a, dim=1)
    log_s = F.log_softmax(z_s, dim=1); log_a = F.log_softmax(z_a, dim=1)
    kl_sa = (p_s * (log_s - log_a)).sum(1)
    kl_as = (p_a * (log_a - log_s)).sum(1)
    d_skl = 0.5 * (kl_sa + kl_as).mean().item()

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
            assert torch.equal(y_s, y_a), "test loaders must produce identical order"

            metrics = function_space_metrics(z_s, z_a)
            metrics.update({"arch": arch, "seed": seed})
            out["runs"].append(metrics)
            print(f"  {arch}/seed={seed}: "
                  f"D_pred={metrics['D_pred']:.4f}  "
                  f"D_SKL={metrics['D_SKL']:.4f}  "
                  f"C_logit={metrics['C_logit']:.4f}", flush=True)

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
