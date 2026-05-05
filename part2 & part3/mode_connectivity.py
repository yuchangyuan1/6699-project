"""
§9 Linear mode connectivity.

For each matched seed and architecture, evaluate

    θ(λ) = (1 - λ) θ_SGD + λ θ_Adam,    λ ∈ {0, 0.1, ..., 1.0}

and report the train and test loss / accuracy along this line. The loss
barrier is

    B = max_λ L̂(θ(λ)) − max{L̂(θ_SGD), L̂(θ_Adam)}.

BatchNorm caveat (§9.4): linear interpolation of trained networks gives
parameters whose batch statistics no longer match the training data. We
recalibrate BN running stats by a forward pass over the training set in
training mode (no parameter update) before evaluating.

Run:
    python mode_connectivity.py
Output:
    results_part2/mode_connectivity.json
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


LAMBDAS = [round(0.1 * i, 2) for i in range(11)]  # 0.0, 0.1, ..., 1.0
SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]
RESULTS_DIR = "./results_part2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                 "mode_connectivity.json"))
    p.add_argument("--bn_calib_batches", type=int, default=50,
                   help="Number of train batches to use for BN recalibration")
    return p.parse_args()


def interpolate_state_dicts(sd_a: dict, sd_b: dict, lam: float) -> dict:
    """Linear interpolation: (1-λ)·a + λ·b for each tensor."""
    out = {}
    for k in sd_a.keys():
        a, b = sd_a[k], sd_b[k]
        if a.dtype.is_floating_point:
            out[k] = (1.0 - lam) * a + lam * b
        else:
            # int buffers (e.g. num_batches_tracked) — keep one side
            out[k] = a.clone()
    return out


@torch.no_grad()
def recalibrate_bn(model: nn.Module, train_loader, device, max_batches: int):
    """
    Reset BN running stats and forward `max_batches` training batches in train
    mode (no grad) to repopulate them. Required after parameter interpolation.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
    model.train()
    n = 0
    for xb, _ in train_loader:
        xb = xb.to(device)
        _ = model(xb)
        n += 1
        if n >= max_batches:
            break
    model.eval()


@torch.no_grad()
def eval_loss_acc(model: nn.Module, loader, device, max_batches=None):
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
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
        "runs": [],
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

            # The matched-seed shuffle order shared by SGD and Adam runs
            train_loader, test_loader = get_loaders(seed)

            run_curves = []
            for lam in LAMBDAS:
                interp_sd = interpolate_state_dicts(sd_sgd, sd_ada, lam)
                model = build_model(arch).to(device)
                model.load_state_dict(interp_sd)

                # BN recalibration (skip at λ=0 and λ=1: the endpoints already
                # have correct BN stats from training)
                if 0.0 < lam < 1.0:
                    recalibrate_bn(model, train_loader, device,
                                   max_batches=args.bn_calib_batches)

                tr_loss, tr_acc = eval_loss_acc(
                    model, train_loader, device, max_batches=80
                )
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
            endpoint_tr = max(run_curves[0]["train_loss"],
                              run_curves[-1]["train_loss"])
            endpoint_te = max(run_curves[0]["test_loss"],
                              run_curves[-1]["test_loss"])
            barrier_train = tr_max - endpoint_tr
            barrier_test  = te_max - endpoint_te

            out["runs"].append({
                "arch": arch, "seed": seed,
                "curves": run_curves,
                "barrier_train": barrier_train,
                "barrier_test":  barrier_test,
            })
            print(f"  {arch}/seed={seed}: B_train={barrier_train:.4f}  "
                  f"B_test={barrier_test:.4f}  "
                  f"min_test_acc={min(c['test_acc'] for c in run_curves):.4f}",
                  flush=True)

    # Aggregate barriers per architecture
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
