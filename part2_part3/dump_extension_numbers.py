"""
After the v2 retrain and run_extended_pipeline.py finish, this prints all the
numerical values that need to be substituted into extension_sections.tex.

Output is a single text block with one section per [PLACEHOLDER] in the LaTeX
template. Values are formatted with stds as `mean ± std`.

Usage:
    python dump_extension_numbers.py
    python dump_extension_numbers.py > extension_numbers.txt
"""
import os
import json
import glob
import numpy as np

RESULTS_DIR = "./results_part2"
SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["MLP", "SmallCNN"]


def fmt(mu, sd, decimals=3):
    return f"${mu:.{decimals}f} \\pm {sd:.{decimals}f}$"


def fmt_int(mu, sd):
    return f"${mu:.1f} \\pm {sd:.1f}$"


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


def section_iv():
    print("=" * 72)
    print("§IV Trajectory Geometry — Table 1 (path length, directness, step norm)")
    print("=" * 72)
    rows = []
    for arch in ARCHS:
        for opt in ["SGD", "Adam"]:
            d = collect_per_run_field(arch, opt, "param_distances")
            P = collect_per_run_field(arch, opt, "final_path_length")
            R = collect_per_run_field(arch, opt, "directness_ratio")
            U = collect_per_run_field(arch, opt, "mean_step_norm")
            if d is None or P is None or R is None or U is None:
                rows.append(f"{arch} & {opt} & MISSING")
                continue
            d_final = d[:, -1]
            row = (f"{arch:<8} & {opt:<4} & "
                   f"{fmt(d_final.mean(), d_final.std(ddof=0), 2)} & "
                   f"{fmt(P.mean(), P.std(ddof=0), 1)} & "
                   f"{fmt(R.mean(), R.std(ddof=0), 4)} & "
                   f"{fmt(U.mean(), U.std(ddof=0), 4)} \\\\")
            rows.append(row)
    print("\n".join(rows))


def section_v():
    print("\n" + "=" * 72)
    print("§V Loss-Matched Geometry — Table 2")
    print("=" * 72)
    p = os.path.join(RESULTS_DIR, "loss_matched_geometry.json")
    if not os.path.exists(p):
        print("MISSING loss_matched_geometry.json"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        e_star = agg["matched_epoch_mean"]
        print(f"\n{arch} (mean matched SGD epoch t* = {e_star:.1f}):")
        print(f"  SGD@t*  : λ_max={fmt(agg['sharp_S_mean'], agg['sharp_S_std'], 2)}, "
              f"tr(H)={fmt(agg['trace_S_mean'], agg['trace_S_std'], 1)}, "
              f"dist={fmt(agg['dist_S_mean'], agg['dist_S_std'], 2)}")
        print(f"  Adam@T  : λ_max={fmt(agg['sharp_A_mean'], agg['sharp_A_std'], 2)}, "
              f"tr(H)={fmt(agg['trace_A_mean'], agg['trace_A_std'], 1)}, "
              f"dist={fmt(agg['dist_A_mean'], agg['dist_A_std'], 2)}")
        print(f"  Function-space @ matched: "
              f"D_pred={fmt(agg['D_pred_mean'], agg['D_pred_std'], 4)}, "
              f"D_SKL={fmt(agg['D_SKL_mean'], agg['D_SKL_std'], 4)}")


def section_vi():
    print("\n" + "=" * 72)
    print("§VI Perturbation Flatness — describe in caption + figure ref")
    print("=" * 72)
    p = os.path.join(RESULTS_DIR, "perturbation_flatness.json")
    if not os.path.exists(p):
        print("MISSING perturbation_flatness.json"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        for opt in ["SGD", "Adam"]:
            key = f"{arch}_{opt}"
            if key not in data["aggregated"]: continue
            agg = data["aggregated"][key]
            print(f"\n{arch}/{opt} (n_seeds={agg['n_seeds']}):")
            for sigma_key, sd in agg["sigmas"].items():
                print(f"  σ={sigma_key:>8}: ΔL̂ = {sd['mean']:+.5f}  (±{sd['std']:.5f})")


def section_vii():
    print("\n" + "=" * 72)
    print("§VII Mode Connectivity — Table 3 (loss barrier)")
    print("=" * 72)
    p = os.path.join(RESULTS_DIR, "mode_connectivity.json")
    if not os.path.exists(p):
        print("MISSING mode_connectivity.json"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch}: B_train = {fmt(agg['barrier_train_mean'], agg['barrier_train_std'], 4)}, "
              f"B_test = {fmt(agg['barrier_test_mean'], agg['barrier_test_std'], 4)}, "
              f"n_seeds = {agg['n_seeds']}")


def section_viii():
    print("\n" + "=" * 72)
    print("§VIII Function-Space Similarity — Table 4")
    print("=" * 72)
    p = os.path.join(RESULTS_DIR, "function_space.json")
    if not os.path.exists(p):
        print("MISSING function_space.json"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch}: "
              f"D_pred={fmt(agg['D_pred_mean'], agg['D_pred_std'], 4)}  "
              f"D_SKL={fmt(agg['D_SKL_mean'], agg['D_SKL_std'], 4)}  "
              f"C_logit={fmt(agg['C_logit_mean'], agg['C_logit_std'], 4)}")


def section_ix():
    print("\n" + "=" * 72)
    print("§IX Representation-Space — Table 5")
    print("=" * 72)
    p = os.path.join(RESULTS_DIR, "representation_cka.json")
    if not os.path.exists(p):
        print("MISSING representation_cka.json"); return
    with open(p) as f: data = json.load(f)
    for arch in ARCHS:
        agg = data["aggregated"].get(arch)
        if agg is None: continue
        print(f"\n{arch}: "
              f"A = {fmt(agg['A_mean'], agg['A_std'], 4)}  "
              f"CKA = {fmt(agg['CKA_mean'], agg['CKA_std'], 4)}")


def main():
    section_iv()
    section_v()
    section_vi()
    section_vii()
    section_viii()
    section_ix()
    print("\nDone. Copy each block into the corresponding [PLACEHOLDER] in "
          "extension_sections.tex.")


if __name__ == "__main__":
    main()
