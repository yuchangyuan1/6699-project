"""
Print every numerical value referenced in the Part II / Part III tables
of the report, formatted as `mean +/- std` over the matched-seed runs.

Usage:
    python dump_numbers.py
    python dump_numbers.py > numbers.txt
"""
import os
import json

import numpy as np


RESULTS_DIR = "./results_part2"
SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]


def fmt(mu, sd, decimals=3):
    return f"${mu:.{decimals}f} \\pm {sd:.{decimals}f}$"


def collect_per_run_field(arch, opt, key):
    vals = []
    for s in SEEDS:
        p = os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{s}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        if key in d:
            vals.append(d[key])
    return np.array(vals) if vals else None


def trajectory():
    print("=" * 72)
    print("Trajectory geometry (path length, directness, step norm)")
    print("=" * 72)
    for arch in ARCHS:
        for opt in ["SGD", "Adam"]:
            d = collect_per_run_field(arch, opt, "param_distances")
            P = collect_per_run_field(arch, opt, "final_path_length")
            R = collect_per_run_field(arch, opt, "directness_ratio")
            U = collect_per_run_field(arch, opt, "mean_step_norm")
            if any(x is None for x in (d, P, R, U)):
                print(f"{arch} & {opt} & MISSING"); continue
            d_final = d[:, -1]
            print(f"{arch:<8} & {opt:<4} & "
                  f"{fmt(d_final.mean(), d_final.std(ddof=0), 2)} & "
                  f"{fmt(P.mean(), P.std(ddof=0), 1)} & "
                  f"{fmt(R.mean(), R.std(ddof=0), 4)} & "
                  f"{fmt(U.mean(), U.std(ddof=0), 4)} \\\\")


def loss_matched():
    print("\n" + "=" * 72); print("Loss-matched geometry"); print("=" * 72)
    p = os.path.join(RESULTS_DIR, "loss_matched_geometry.json")
    if not os.path.exists(p): print("MISSING loss_matched_geometry.json"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch} (mean matched SGD epoch t* = {agg['matched_epoch_mean']:.1f}):")
        print(f"  SGD@t*: lambda_max={fmt(agg['sharp_S_mean'], agg['sharp_S_std'], 2)}, "
              f"tr(H)={fmt(agg['trace_S_mean'], agg['trace_S_std'], 1)}, "
              f"dist={fmt(agg['dist_S_mean'], agg['dist_S_std'], 2)}")
        print(f"  Adam@T: lambda_max={fmt(agg['sharp_A_mean'], agg['sharp_A_std'], 2)}, "
              f"tr(H)={fmt(agg['trace_A_mean'], agg['trace_A_std'], 1)}, "
              f"dist={fmt(agg['dist_A_mean'], agg['dist_A_std'], 2)}")
        print(f"  Function-space @ matched: "
              f"D_pred={fmt(agg['D_pred_mean'], agg['D_pred_std'], 4)}, "
              f"D_SKL={fmt(agg['D_SKL_mean'], agg['D_SKL_std'], 4)}")


def perturbation():
    print("\n" + "=" * 72); print("Perturbation flatness"); print("=" * 72)
    p = os.path.join(RESULTS_DIR, "perturbation_flatness.json")
    if not os.path.exists(p): print("MISSING"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        for opt in ["SGD", "Adam"]:
            key = f"{arch}_{opt}"
            if key not in data["aggregated"]: continue
            agg = data["aggregated"][key]
            print(f"\n{arch}/{opt} (n_seeds={agg['n_seeds']}):")
            for sigma_key, sd in agg["sigmas"].items():
                print(f"  sigma={sigma_key:>8}: DeltaL = {sd['mean']:+.5f}  (+/-{sd['std']:.5f})")


def mode_connectivity():
    print("\n" + "=" * 72); print("Mode connectivity (loss barrier)"); print("=" * 72)
    p = os.path.join(RESULTS_DIR, "mode_connectivity.json")
    if not os.path.exists(p): print("MISSING"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch}: B_train = {fmt(agg['barrier_train_mean'], agg['barrier_train_std'], 4)}, "
              f"B_test = {fmt(agg['barrier_test_mean'], agg['barrier_test_std'], 4)}, "
              f"n_seeds = {agg['n_seeds']}")


def function_space():
    print("\n" + "=" * 72); print("Function-space similarity"); print("=" * 72)
    p = os.path.join(RESULTS_DIR, "function_space.json")
    if not os.path.exists(p): print("MISSING"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch}: "
              f"D_pred={fmt(agg['D_pred_mean'], agg['D_pred_std'], 4)}  "
              f"D_SKL={fmt(agg['D_SKL_mean'], agg['D_SKL_std'], 4)}  "
              f"C_logit={fmt(agg['C_logit_mean'], agg['C_logit_std'], 4)}")


def representation():
    print("\n" + "=" * 72); print("Representation alignment"); print("=" * 72)
    p = os.path.join(RESULTS_DIR, "representation_cka.json")
    if not os.path.exists(p): print("MISSING"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch}: "
              f"A = {fmt(agg['A_mean'], agg['A_std'], 4)}  "
              f"CKA = {fmt(agg['CKA_mean'], agg['CKA_std'], 4)}")


def main():
    trajectory()
    loss_matched()
    perturbation()
    mode_connectivity()
    function_space()
    representation()


if __name__ == "__main__":
    main()
