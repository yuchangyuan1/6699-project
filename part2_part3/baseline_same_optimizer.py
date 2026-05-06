"""
Same-optimizer cross-seed baseline for §VII–IX.

For each (architecture, optimizer), enumerate the C(5,2)=10 distinct seed
pairs (i, j) and recompute the same metrics that the cross-optimizer scripts
report on matched seeds:

    Function-space (function_space.py):
        D_pred, D_SKL, C_logit
    Representation (representation_cka.py):
        feature_kernel_alignment (A), linear_cka (CKA)
    Mode connectivity (mode_connectivity.py):
        barrier_train, barrier_test  (with BN recalibration on interior λ)

This is the falsification baseline: if same-opt cross-seed values come out
close to the cross-opt matched-seed values reported in §VII–IX, the
optimizer-induced effect there is mostly seed noise, not optimizer choice.

Run:
    python baseline_same_optimizer.py
Output:
    results_part2/baseline_same_optimizer.json

Notes:
- For a same-opt pair (seed_i, seed_j), the BN recalibration loader and
  test loader are taken from get_loaders(seed_i). The metrics are symmetric
  in the two models, so the choice of which seed drives the loader is
  conventional. The same convention is used for all (arch, opt, i, j).
- recalibrate_bn is a no-op when the model has no BN modules, so the MLP
  case simply skips BN handling.
"""
import os
import json
import time
import argparse
import itertools
import numpy as np
import torch

from model_io import build_model, load_final_model
from data_utils import get_loaders
from function_space import collect_logits, function_space_metrics
from representation_cka import (
    collect_test_inputs,
    extract_features,
    feature_kernel_alignment,
    linear_cka,
)
from mode_connectivity import (
    LAMBDAS,
    interpolate_state_dicts,
    recalibrate_bn,
    eval_loss_acc,
)


SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]
OPTS = ["SGD", "Adam"]
RESULTS_DIR = "./results_part2"
N_CKA_SAMPLES = 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                 "baseline_same_optimizer.json"))
    p.add_argument("--bn_calib_batches", type=int, default=50)
    p.add_argument("--n_cka", type=int, default=N_CKA_SAMPLES)
    return p.parse_args()


def compute_pair(arch, opt, seed_i, seed_j, device, bn_calib_batches, n_cka):
    """Compute all seven metrics for a same-(arch,opt) seed pair."""
    m_i = load_final_model(arch, opt, seed_i, device)
    m_j = load_final_model(arch, opt, seed_j, device)

    train_loader, test_loader = get_loaders(seed_i)

    # 1. Function-space metrics on the test set.
    z_i, y_i = collect_logits(m_i, test_loader, device)
    z_j, y_j = collect_logits(m_j, test_loader, device)
    assert torch.equal(y_i, y_j), "test loaders must produce identical order"
    fs = function_space_metrics(z_i, z_j)

    # 2. Representation alignment on a fixed test subset.
    x, _ = collect_test_inputs(test_loader, n_cka, device)
    F_i = extract_features(m_i, x)
    F_j = extract_features(m_j, x)
    A   = feature_kernel_alignment(F_i, F_j)
    cka = linear_cka(F_i, F_j)

    # 3. Linear mode connectivity along θ(λ) = (1-λ) θ_i + λ θ_j.
    sd_i = {k: v.detach().clone() for k, v in m_i.state_dict().items()}
    sd_j = {k: v.detach().clone() for k, v in m_j.state_dict().items()}
    curves = []
    for lam in LAMBDAS:
        interp_sd = interpolate_state_dicts(sd_i, sd_j, lam)
        model = build_model(arch).to(device)
        model.load_state_dict(interp_sd)
        if 0.0 < lam < 1.0:
            recalibrate_bn(model, train_loader, device,
                           max_batches=bn_calib_batches)
        tr_loss, tr_acc = eval_loss_acc(
            model, train_loader, device, max_batches=80
        )
        te_loss, te_acc = eval_loss_acc(model, test_loader, device)
        curves.append({
            "lambda":     lam,
            "train_loss": tr_loss,
            "train_acc":  tr_acc,
            "test_loss":  te_loss,
            "test_acc":   te_acc,
        })

    tr_max = max(c["train_loss"] for c in curves)
    te_max = max(c["test_loss"]  for c in curves)
    endpoint_tr = max(curves[0]["train_loss"], curves[-1]["train_loss"])
    endpoint_te = max(curves[0]["test_loss"],  curves[-1]["test_loss"])
    barrier_train = tr_max - endpoint_tr
    barrier_test  = te_max - endpoint_te

    return {
        "arch":   arch,
        "opt":    opt,
        "seed_i": seed_i,
        "seed_j": seed_j,
        "D_pred":  fs["D_pred"],
        "D_SKL":   fs["D_SKL"],
        "C_logit": fs["C_logit"],
        "feature_kernel_alignment": A,
        "linear_cka":               cka,
        "barrier_train": barrier_train,
        "barrier_test":  barrier_test,
        "curves":        curves,
    }


def aggregate(runs):
    """Group runs by (arch, opt) and report mean / std of every metric."""
    keys = ["D_pred", "D_SKL", "C_logit",
            "feature_kernel_alignment", "linear_cka",
            "barrier_train", "barrier_test"]
    bucket = {}
    for r in runs:
        cell = f"{r['arch']}_{r['opt']}"
        bucket.setdefault(cell, {k: [] for k in keys})
        for k in keys:
            bucket[cell][k].append(r[k])
    out = {}
    for cell, vals in bucket.items():
        agg = {f"{k}_mean": float(np.mean(vals[k])) for k in keys}
        agg.update({f"{k}_std": float(np.std(vals[k], ddof=0)) for k in keys})
        # Convenience aliases for the most-cited metrics in §VII–IX.
        agg["A_mean"]   = agg["feature_kernel_alignment_mean"]
        agg["A_std"]    = agg["feature_kernel_alignment_std"]
        agg["CKA_mean"] = agg["linear_cka_mean"]
        agg["CKA_std"]  = agg["linear_cka_std"]
        agg["n_pairs"]  = len(vals[keys[0]])
        out[cell] = agg
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out = {
        "config": {
            "lambdas":           LAMBDAS,
            "bn_calib_batches":  args.bn_calib_batches,
            "n_pairs_per_cell":  10,
            "n_cka_samples":     args.n_cka,
        },
        "runs": [],
    }
    t0 = time.time()
    pair_total = len(ARCHS) * len(OPTS) * 10
    pair_idx = 0

    for arch in ARCHS:
        for opt in OPTS:
            for seed_i, seed_j in itertools.combinations(SEEDS, 2):
                pair_idx += 1
                tag = f"[{pair_idx:>2d}/{pair_total}] {arch}/{opt} ({seed_i},{seed_j})"
                try:
                    r = compute_pair(
                        arch, opt, seed_i, seed_j, device,
                        bn_calib_batches=args.bn_calib_batches,
                        n_cka=args.n_cka,
                    )
                except Exception as e:
                    print(f"{tag}: SKIP — {type(e).__name__}: {e}", flush=True)
                    continue
                out["runs"].append(r)
                print(f"{tag}: D_pred={r['D_pred']:.4f}  "
                      f"CKA={r['linear_cka']:.4f}  "
                      f"B_train={r['barrier_train']:.4f}  "
                      f"B_test={r['barrier_test']:.4f}",
                      flush=True)

    out["aggregated"] = aggregate(out["runs"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Wrote {len(out['runs'])}/{pair_total} pairs to {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
