"""
§7.1 Loss-matched geometry control.

Fixed-epoch comparisons can confound geometry differences with training-loss
differences. For each (arch, seed) we therefore:

    1. Read Adam's final training loss L̂_A from `_per_epoch` arrays.
    2. Search the SGD trajectory for epoch
           t* = argmin_t |L̂_S(t) - L̂_A|
    3. Load SGD's epoch-`t*` checkpoint and Adam's final checkpoint.
    4. Compare Hessian λ_max, trace, parameter distance from init,
       function-space disagreement, and feature-kernel alignment.

If Adam still differs geometrically under loss matching, the geometry change
is not merely a final-loss artifact.

Run:
    python loss_matched_geometry.py
Output:
    results_part2/loss_matched_geometry.json
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_io import (build_model, load_final_model, load_epoch_model,
                      RESULTS_DIR, CKPT_SUBDIR)
from data_utils import get_loaders
from geometry import (sharpness_power_iter, hessian_trace_hutchinson,
                      param_distance)
from models import get_flat_params


SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                 "loss_matched_geometry.json"))
    p.add_argument("--sharp_steps", type=int, default=100)
    p.add_argument("--hess_probe", type=int, default=30)
    p.add_argument("--hess_samples", type=int, default=512)
    return p.parse_args()


def find_loss_matched_epoch(sgd_losses, adam_final_loss):
    """
    Return the SGD epoch (1-indexed) closest in train-loss to Adam's final loss.
    `sgd_losses` is the per-epoch list, so the value at index e-1 is the
    average training loss in epoch e.
    """
    diffs = [abs(L - adam_final_loss) for L in sgd_losses]
    e_star = int(np.argmin(diffs)) + 1   # 1-indexed
    return e_star, sgd_losses[e_star - 1]


@torch.no_grad()
def collect_test_logits(model, loader, device):
    model.eval()
    Z, Y = [], []
    for xb, yb in loader:
        Z.append(model(xb.to(device)).cpu())
        Y.append(yb)
    return torch.cat(Z), torch.cat(Y)


def function_space(z1, z2):
    p1 = F.softmax(z1, dim=1); p2 = F.softmax(z2, dim=1)
    l1 = F.log_softmax(z1, dim=1); l2 = F.log_softmax(z2, dim=1)
    d_pred = (z1.argmax(1) != z2.argmax(1)).float().mean().item()
    skl = 0.5 * ((p1 * (l1 - l2)).sum(1) + (p2 * (l2 - l1)).sum(1)).mean().item()
    return d_pred, skl


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out = {"runs": []}
    t0 = time.time()

    for arch in ARCHS:
        for seed in SEEDS:
            sgd_json_path  = os.path.join(RESULTS_DIR, f"{arch}_SGD_seed{seed}.json")
            adam_json_path = os.path.join(RESULTS_DIR, f"{arch}_Adam_seed{seed}.json")
            if not (os.path.exists(sgd_json_path) and os.path.exists(adam_json_path)):
                print(f"SKIP {arch}/seed={seed}: missing JSON", flush=True)
                continue
            with open(sgd_json_path)  as f: sgd_d  = json.load(f)
            with open(adam_json_path) as f: adam_d = json.load(f)
            if "train_loss_per_epoch" not in sgd_d:
                print(f"SKIP {arch}/seed={seed}: v1 JSON without train_loss_per_epoch — "
                      "re-run train_part2.py with --force first", flush=True)
                continue

            adam_final_loss = adam_d["train_loss_per_epoch"][-1]
            sgd_losses      = sgd_d["train_loss_per_epoch"]
            e_star, matched_loss = find_loss_matched_epoch(sgd_losses, adam_final_loss)
            print(f"  {arch}/seed={seed}: Adam final loss={adam_final_loss:.4f}  "
                  f"matched SGD epoch={e_star} (loss={matched_loss:.4f})",
                  flush=True)

            # Load matched SGD checkpoint
            try:
                m_sgd_t = load_epoch_model(arch, "SGD",  seed, e_star, device)
                m_ada   = load_final_model(arch, "Adam", seed, device)
                m_sgd_T = load_final_model(arch, "SGD",  seed, device)  # baseline
            except FileNotFoundError as e:
                print(f"  SKIP: missing checkpoint: {e}", flush=True)
                continue

            train_loader, test_loader = get_loaders(seed)

            # Hessian on the matched SGD checkpoint and Adam final
            sharp_sgd_t = sharpness_power_iter(m_sgd_t, train_loader, device,
                n_steps=args.sharp_steps, max_samples=args.hess_samples)
            trace_sgd_t = hessian_trace_hutchinson(m_sgd_t, train_loader, device,
                n_samples=args.hess_probe, max_samples=args.hess_samples)
            sharp_ada   = adam_d["final_sharpness"]
            trace_ada   = adam_d["final_trace"]

            # Distance from init
            init_flat   = torch.from_numpy(np.load(os.path.join(
                RESULTS_DIR, CKPT_SUBDIR,
                f"{arch}_SGD_seed{seed}", "epoch_00.npy")))
            sgd_t_flat  = get_flat_params(m_sgd_t).cpu()
            ada_flat    = get_flat_params(m_ada).cpu()
            d_sgd_t     = param_distance(sgd_t_flat, init_flat)
            d_ada       = param_distance(ada_flat,   init_flat)

            # Function-space disagreement (loss-matched)
            z_sgd_t, _ = collect_test_logits(m_sgd_t, test_loader, device)
            z_ada,   _ = collect_test_logits(m_ada,   test_loader, device)
            d_pred_lm, d_skl_lm = function_space(z_sgd_t, z_ada)

            entry = {
                "arch": arch, "seed": seed,
                "adam_final_loss":     adam_final_loss,
                "sgd_matched_epoch":   e_star,
                "sgd_matched_loss":    matched_loss,
                "loss_match_residual": abs(matched_loss - adam_final_loss),
                # Hessian comparison (loss-matched)
                "sharpness_sgd_at_t_star": sharp_sgd_t,
                "sharpness_adam_final":    sharp_ada,
                "trace_sgd_at_t_star":     trace_sgd_t,
                "trace_adam_final":        trace_ada,
                # Distance comparison
                "dist_sgd_at_t_star":      d_sgd_t,
                "dist_adam_final":         d_ada,
                # Function-space comparison at matched loss
                "D_pred_loss_matched":     d_pred_lm,
                "D_SKL_loss_matched":      d_skl_lm,
            }
            out["runs"].append(entry)
            print(f"    SGD@t*: λ_max={sharp_sgd_t:.2f}  tr={trace_sgd_t:.2f}  "
                  f"d={d_sgd_t:.2f}     "
                  f"Adam: λ_max={sharp_ada:.2f}  tr={trace_ada:.2f}  d={d_ada:.2f}",
                  flush=True)

    # Aggregate per architecture
    agg = {}
    for r in out["runs"]:
        d = agg.setdefault(r["arch"],
            {"sharp_S": [], "sharp_A": [],
             "trace_S": [], "trace_A": [],
             "dist_S":  [], "dist_A":  [],
             "D_pred":  [], "D_SKL":   [],
             "matched_epochs": []})
        d["sharp_S"].append(r["sharpness_sgd_at_t_star"])
        d["sharp_A"].append(r["sharpness_adam_final"])
        d["trace_S"].append(r["trace_sgd_at_t_star"])
        d["trace_A"].append(r["trace_adam_final"])
        d["dist_S"].append(r["dist_sgd_at_t_star"])
        d["dist_A"].append(r["dist_adam_final"])
        d["D_pred"].append(r["D_pred_loss_matched"])
        d["D_SKL"].append(r["D_SKL_loss_matched"])
        d["matched_epochs"].append(r["sgd_matched_epoch"])
    out["aggregated"] = {
        a: {
            "n_seeds": len(v["sharp_S"]),
            "matched_epoch_mean": float(np.mean(v["matched_epochs"])),
            **{f"{k}_mean": float(np.mean(v[k])) for k in
               ["sharp_S", "sharp_A", "trace_S", "trace_A",
                "dist_S",  "dist_A",  "D_pred",  "D_SKL"]},
            **{f"{k}_std":  float(np.std(v[k], ddof=0)) for k in
               ["sharp_S", "sharp_A", "trace_S", "trace_A",
                "dist_S",  "dist_A",  "D_pred",  "D_SKL"]},
        } for a, v in agg.items()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
