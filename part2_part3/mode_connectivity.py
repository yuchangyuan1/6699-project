"""
Linear mode connectivity between matched-seed SGD and Adam endpoints.

For lambda in {0, 0.1, ..., 1.0}, evaluate train and test loss along
    theta(lambda) = (1 - lambda) * theta_SGD + lambda * theta_Adam.
BatchNorm running stats are recalibrated by a forward pass over a few train
batches at each interior lambda. Loss barrier is reported per (arch, seed).

Output: results_part2/mode_connectivity.json
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

from model_io import build_model, load_final_model
from data_utils import get_loaders


LAMBDAS = [round(0.1 * i, 2) for i in range(11)]
SEEDS   = [42, 123, 456, 789, 2024]
ARCHS   = ["MLP", "SmallCNN"]
RESULTS_DIR = "./results_part2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "mode_connectivity.json"))
    p.add_argument("--bn_calib_batches", type=int, default=50)
    return p.parse_args()


def interpolate_state_dicts(sd_a, sd_b, lam):
    """(1 - lam) * a + lam * b for floating tensors; integer buffers kept from a."""
    out = {}
    for k in sd_a.keys():
        a, b = sd_a[k], sd_b[k]
        if a.dtype.is_floating_point:
            out[k] = (1.0 - lam) * a + lam * b
        else:
            out[k] = a.clone()
    return out


@torch.no_grad()
def recalibrate_bn(model, train_loader, device, max_batches):
    """Reset BN running stats and forward `max_batches` train batches in train mode."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
    model.train()
    n = 0
    for xb, _ in train_loader:
        _ = model(xb.to(device))
        n += 1
        if n >= max_batches:
            break
    model.eval()


@torch.no_grad()
def eval_loss_acc(model, loader, device, max_batches=None):
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = total = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += crit(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out = {
        "config": {"lambdas": LAMBDAS, "bn_calib_batches": args.bn_calib_batches},
        "runs":   [],
    }

    t0 = time.time()
    for arch in ARCHS:
        for seed in SEEDS:
            try:
                m_sgd = load_final_model(arch, "SGD", seed, device)
                m_ada = load_final_model(arch, "Adam", seed, device)
            except FileNotFoundError as e:
                print(f"SKIP {arch}/seed={seed}: {e}", flush=True)
                continue

            sd_sgd = {k: v.detach().clone() for k, v in m_sgd.state_dict().items()}
            sd_ada = {k: v.detach().clone() for k, v in m_ada.state_dict().items()}
            train_loader, test_loader = get_loaders(seed)

            run_curves = []
            for lam in LAMBDAS:
                interp_sd = interpolate_state_dicts(sd_sgd, sd_ada, lam)
                model = build_model(arch).to(device)
                model.load_state_dict(interp_sd)
                if 0.0 < lam < 1.0:
                    recalibrate_bn(model, train_loader, device,
                                   max_batches=args.bn_calib_batches)
                tr_loss, tr_acc = eval_loss_acc(model, train_loader, device, max_batches=80)
                te_loss, te_acc = eval_loss_acc(model, test_loader, device)
                run_curves.append({
                    "lambda":     lam,
                    "train_loss": tr_loss,
                    "train_acc":  tr_acc,
                    "test_loss":  te_loss,
                    "test_acc":   te_acc,
                })

            tr_max = max(c["train_loss"] for c in run_curves)
            te_max = max(c["test_loss"]  for c in run_curves)
            endpoint_tr = max(run_curves[0]["train_loss"],  run_curves[-1]["train_loss"])
            endpoint_te = max(run_curves[0]["test_loss"],   run_curves[-1]["test_loss"])
            out["runs"].append({
                "arch": arch, "seed": seed,
                "curves": run_curves,
                "barrier_train": tr_max - endpoint_tr,
                "barrier_test":  te_max - endpoint_te,
            })
            print(f"  {arch}/seed={seed}: B_train={tr_max-endpoint_tr:.4f}  "
                  f"B_test={te_max-endpoint_te:.4f}", flush=True)

    agg = {}
    for r in out["runs"]:
        agg.setdefault(r["arch"], {"barrier_train": [], "barrier_test": []})
        agg[r["arch"]]["barrier_train"].append(r["barrier_train"])
        agg[r["arch"]]["barrier_test"].append(r["barrier_test"])
    out["aggregated"] = {
        a: {
            "barrier_train_mean": float(np.mean(v["barrier_train"])),
            "barrier_train_std":  float(np.std(v["barrier_train"], ddof=0)),
            "barrier_test_mean":  float(np.mean(v["barrier_test"])),
            "barrier_test_std":   float(np.std(v["barrier_test"], ddof=0)),
            "n_seeds":            len(v["barrier_train"]),
        } for a, v in agg.items()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
