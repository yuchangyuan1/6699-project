"""
Part II: Solution Geometry and Implicit Bias
============================================
Trains MLP and SmallCNN with SGD and Adam across 5 matched seeds,
and records all Part II metrics.

Runtime guide:
  GPU (Colab/server): ~25 minutes for all 20 runs
  CPU only (local):   ~4 hours  (use FAST_TEST=True for a ~5-min smoke test)

Usage:
    python train_part2.py               # full run
    python train_part2.py --fast        # reduced settings, ~5 min on CPU

Outputs saved to ./results_part2/
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from models import MLP, SmallCNN, get_flat_params
from data_utils import get_loaders
from geometry import (
    sharpness_power_iter,
    hessian_trace_hutchinson,
    param_distance,
)

# ── Full-run hyperparameters (Table 1 in the paper) ─────────────────────────
SEEDS       = [42, 123, 456, 789, 2024]
EPOCHS      = 30
BATCH_SIZE  = 128
SGD_LR      = 0.01
SGD_MOM     = 0.9
ADAM_LR     = 0.001
ADAM_B1     = 0.9
ADAM_B2     = 0.999
DATA_ROOT   = "./data"
RESULTS_DIR = "./results_part2"
CKPT_SUBDIR = "checkpoints"  # epoch-level snapshots for loss-matched analysis

# Hessian estimation settings (full run)
SHARP_STEPS    = 100    # power iteration steps
SHARP_TOL      = 1e-4
HESS_PROBE     = 30     # Hutchinson probe vectors
HESS_SAMPLES   = 512    # mini-batch size for Hessian

ARCHITECTURES = {"MLP": MLP, "SmallCNN": SmallCNN}

# ── Fast-test overrides ──────────────────────────────────────────────────────
FAST_SEEDS      = [42]
FAST_EPOCHS     = 5
FAST_SHARP_STEPS= 20
FAST_HESS_PROBE = 5
FAST_HESS_SAMP  = 128
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="Quick smoke test with reduced settings")
    p.add_argument("--arch", default=None,
                   help="Only run one architecture: MLP or SmallCNN")
    p.add_argument("--opt", default=None,
                   help="Only run one optimizer: SGD or Adam")
    p.add_argument("--seed", type=int, default=None,
                   help="Only run one seed")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output JSON exists (needed when JSON is missing v2 fields like path_length_per_epoch)")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_optimizer(name: str, model: nn.Module):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=SGD_LR,
                               momentum=SGD_MOM, weight_decay=0)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=ADAM_LR,
                                betas=(ADAM_B1, ADAM_B2), weight_decay=0)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
    return correct / total


def train_one_run(
    arch_name: str,
    opt_name: str,
    seed: int,
    device: torch.device,
    cfg: dict,
) -> dict:
    """
    Full training run for one (architecture, optimizer, seed) triple.

    cfg keys: epochs, sharp_steps, sharp_tol, hess_probe, hess_samples

    Returns a dict with all Part II metrics.
    """
    epochs       = cfg["epochs"]
    sharp_steps  = cfg["sharp_steps"]
    sharp_tol    = cfg["sharp_tol"]
    hess_probe   = cfg["hess_probe"]
    hess_samples = cfg["hess_samples"]

    set_seed(seed)

    model = ARCHITECTURES[arch_name]().to(device)
    init_flat = get_flat_params(model).clone()  # theta_0

    optimizer = build_optimizer(opt_name, model)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_loaders(seed, BATCH_SIZE, DATA_ROOT)

    # Per-epoch metrics
    param_distances        = []   # ||theta_t - theta_0||_2 per epoch
    grad_norms             = []   # mean ||g_t||_2 per epoch
    train_acc_curve        = []   # train accuracy per epoch
    test_acc_curve         = []   # test accuracy per epoch
    train_loss_per_epoch   = []   # mean train loss per epoch (needed for loss-matched §7.1)
    path_length_per_epoch  = []   # cumulative path length P_T at epoch end (§6.2)
    step_norm_per_epoch    = []   # mean ||Δθ_t|| within each epoch (§6.3)

    # Per-step path-length accumulator
    step_norm_sum   = 0.0   # cumulative Σ ||Δθ_t||  (path length)
    step_count_total = 0

    # Per-epoch parameter checkpoint dir (for loss-matched control §7.1)
    ckpt_dir = os.path.join(RESULTS_DIR, CKPT_SUBDIR, f"{arch_name}_{opt_name}_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save θ_0 once (epoch 0): both flat .npy and full state_dict .pt
    # state_dict is required for downstream eval scripts because it includes
    # BatchNorm running_mean / running_var buffers.
    np.save(os.path.join(ckpt_dir, "epoch_00.npy"),
            init_flat.cpu().numpy().astype(np.float32))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               os.path.join(ckpt_dir, "epoch_00.pt"))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_gnorms = []
        epoch_step_norms = []
        epoch_loss   = 0.0
        n_batches    = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()

            # Collect gradient norm before stepping (essentially free)
            gnorm = torch.cat([
                p.grad.detach().flatten()
                for p in model.parameters()
                if p.grad is not None
            ]).norm().item()
            epoch_gnorms.append(gnorm)
            epoch_loss += loss.item()
            n_batches  += 1

            # Snapshot parameters before optimizer.step() for ||Δθ_t||
            prev_flat = get_flat_params(model)
            optimizer.step()
            new_flat  = get_flat_params(model)
            step_norm = (new_flat - prev_flat).norm().item()
            step_norm_sum += step_norm
            epoch_step_norms.append(step_norm)
            step_count_total += 1

        # ── Epoch metrics ─────────────────────────────────────────────────
        flat_now = get_flat_params(model)
        dist     = param_distance(flat_now, init_flat)
        gnorm_avg     = float(np.mean(epoch_gnorms))
        step_norm_avg = float(np.mean(epoch_step_norms))
        avg_train_loss = epoch_loss / n_batches

        param_distances.append(dist)
        grad_norms.append(gnorm_avg)
        train_loss_per_epoch.append(avg_train_loss)
        path_length_per_epoch.append(step_norm_sum)
        step_norm_per_epoch.append(step_norm_avg)

        # Save epoch-end snapshot for loss-matched analysis (§7.1)
        # Both the flat .npy (params only) and the full .pt (state_dict with BN buffers)
        np.save(os.path.join(ckpt_dir, f"epoch_{epoch:02d}.npy"),
                flat_now.cpu().numpy().astype(np.float32))
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt"))

        # Accuracy evaluated every epoch (needed for curves)
        tr_acc = evaluate(model, train_loader, device)
        te_acc = evaluate(model, test_loader,  device)
        train_acc_curve.append(tr_acc)
        test_acc_curve.append(te_acc)

        print(f"  [{arch_name}/{opt_name}/seed={seed}] "
              f"ep {epoch:2d}/{epochs}  "
              f"loss={avg_train_loss:.4f}  "
              f"dist={dist:.2f}  P={step_norm_sum:.2f}  "
              f"gnorm={gnorm_avg:.4f}  "
              f"tr={tr_acc:.4f}  te={te_acc:.4f}",
              flush=True)

    # ── Final Hessian metrics ─────────────────────────────────────────────
    print(f"  Computing λ_max  [{arch_name}/{opt_name}/seed={seed}]...", flush=True)
    sharpness = sharpness_power_iter(
        model, train_loader, device,
        n_steps=sharp_steps, tol=sharp_tol, max_samples=hess_samples,
    )

    print(f"  Computing tr(H)  [{arch_name}/{opt_name}/seed={seed}]...", flush=True)
    trace = hessian_trace_hutchinson(
        model, train_loader, device,
        n_samples=hess_probe, max_samples=hess_samples,
    )

    print(f"  λ_max={sharpness:.2f}  tr(H)={trace:.2f}", flush=True)

    final_flat = get_flat_params(model).cpu().numpy()

    final_dist        = float(param_distances[-1])
    final_path_length = float(step_norm_sum)
    directness_ratio  = (final_dist / final_path_length) if final_path_length > 0 else 0.0

    return {
        "arch":            arch_name,
        "optimizer":       opt_name,
        "seed":            seed,
        # Existing per-epoch curves
        "param_distances": param_distances,
        "grad_norms":      grad_norms,
        "train_acc_curve": train_acc_curve,
        "test_acc_curve":  test_acc_curve,
        # New per-epoch curves (Plan B additions)
        "train_loss_per_epoch":  train_loss_per_epoch,
        "path_length_per_epoch": path_length_per_epoch,
        "step_norm_per_epoch":   step_norm_per_epoch,
        # Final scalars
        "final_sharpness":   float(sharpness),
        "final_trace":       float(trace),
        "final_path_length": final_path_length,
        "mean_step_norm":    final_path_length / max(step_count_total, 1),
        "directness_ratio":  directness_ratio,
        "total_steps":       step_count_total,
        # Reference to checkpoint directory
        "checkpoint_dir":    ckpt_dir,
        "_final_flat":       final_flat,    # removed before JSON, saved as .npy
    }


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Build config
    if args.fast:
        print("FAST TEST MODE (reduced settings)", flush=True)
        cfg = {
            "epochs":       FAST_EPOCHS,
            "sharp_steps":  FAST_SHARP_STEPS,
            "sharp_tol":    1e-3,
            "hess_probe":   FAST_HESS_PROBE,
            "hess_samples": FAST_HESS_SAMP,
        }
        seeds    = FAST_SEEDS
        archs    = list(ARCHITECTURES)
        opts     = ["SGD", "Adam"]
    else:
        cfg = {
            "epochs":       EPOCHS,
            "sharp_steps":  SHARP_STEPS,
            "sharp_tol":    SHARP_TOL,
            "hess_probe":   HESS_PROBE,
            "hess_samples": HESS_SAMPLES,
        }
        seeds = SEEDS
        archs = list(ARCHITECTURES)
        opts  = ["SGD", "Adam"]

    # Apply CLI filters
    if args.arch:  archs = [args.arch]
    if args.opt:   opts  = [args.opt]
    if args.seed:  seeds = [args.seed]

    t_global = time.time()
    completed = 0
    total_runs = len(archs) * len(opts) * len(seeds)

    for arch_name in archs:
        for opt_name in opts:
            for seed in seeds:
                out_json = os.path.join(RESULTS_DIR,
                    f"{arch_name}_{opt_name}_seed{seed}.json")
                out_npy  = out_json.replace(".json", "_params.npy")

                # Skip only if file exists AND has the new v2 fields.
                # --force always overrides.
                if os.path.exists(out_json) and not args.force:
                    try:
                        with open(out_json) as _f:
                            _existing = json.load(_f)
                        if "path_length_per_epoch" in _existing:
                            print(f"Skip (exists w/ v2 fields): {out_json}", flush=True)
                            completed += 1
                            continue
                        else:
                            print(f"Re-running (v1 JSON missing path_length_per_epoch): {out_json}", flush=True)
                    except Exception:
                        print(f"Re-running (cannot parse existing JSON): {out_json}", flush=True)

                print(f"\n{'='*62}", flush=True)
                print(f"  RUN {completed+1}/{total_runs}: "
                      f"{arch_name} | {opt_name} | seed={seed}", flush=True)
                print(f"{'='*62}", flush=True)
                t0 = time.time()

                result = train_one_run(arch_name, opt_name, seed, device, cfg)

                # Save parameter vector separately (large array)
                np.save(out_npy, result.pop("_final_flat").astype(np.float32))
                # Also save the final state_dict (.pt) at the top level for convenience
                # — it is identical to ckpt_dir/epoch_{EPOCHS:02d}.pt
                final_pt_src = os.path.join(result["checkpoint_dir"],
                    f"epoch_{cfg['epochs']:02d}.pt")
                final_pt_dst = out_json.replace(".json", "_state.pt")
                if os.path.exists(final_pt_src):
                    import shutil
                    shutil.copyfile(final_pt_src, final_pt_dst)

                with open(out_json, "w") as f:
                    json.dump(result, f, indent=2)

                elapsed = time.time() - t0
                completed += 1
                print(f"  Saved: {out_json}  ({elapsed:.0f}s)", flush=True)

    total_time = time.time() - t_global
    print(f"\nAll {completed}/{total_runs} runs done in {total_time/60:.1f} min.",
          flush=True)
    print(f"Results in: {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
