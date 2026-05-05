"""
Part III figures — generated from results_part2/ JSON files.

Produces (saved to figures_part3/):
  fig_test_acc_mlp.pdf/png          — test accuracy curves, MLP
  fig_test_acc_cnn.pdf/png          — test accuracy curves, CNN
  fig_gap_mlp.pdf/png               — train-test gap curves, MLP
  fig_gap_cnn.pdf/png               — train-test gap curves, CNN
  fig_sharpness_gap_mlp.pdf/png     — sharpness vs gap scatter, MLP
  fig_sharpness_gap_cnn.pdf/png     — sharpness vs gap scatter, CNN

LaTeX refs in the report use figures_part3/{name}.pdf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SEEDS = [42, 123, 456, 789, 2024]
RESULTS = "results_part2"
OUT_DIR = "figures_part3"
os.makedirs(OUT_DIR, exist_ok=True)

COLOR = {"SGD": "#2166ac", "Adam": "#d6604d"}
LS    = {"SGD": "-",       "Adam": "--"}

# ── load ─────────────────────────────────────────────────────────────────────
def load_runs(arch):
    data = {}
    for opt in ("SGD", "Adam"):
        runs = []
        for seed in SEEDS:
            fname = os.path.join(RESULTS, f"{arch}_{opt}_seed{seed}.json")
            with open(fname) as f:
                runs.append(json.load(f))
        data[opt] = runs
    return data

# ── helpers ──────────────────────────────────────────────────────────────────
def mean_std(matrix):
    arr = np.array(matrix)          # (n_seeds, n_epochs)
    return arr.mean(0), arr.std(0)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf/png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 & 2  — Test accuracy curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_test_acc(arch, runs, tag):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    epochs = np.arange(1, len(runs["SGD"][0]["test_acc_curve"]) + 1)

    for opt in ("SGD", "Adam"):
        curves = [r["test_acc_curve"] for r in runs[opt]]
        mu, sd = mean_std(curves)
        mu_pct, sd_pct = mu * 100, sd * 100
        ax.plot(epochs, mu_pct, color=COLOR[opt], ls=LS[opt],
                lw=1.8, label=opt)
        ax.fill_between(epochs, mu_pct - sd_pct, mu_pct + sd_pct,
                        color=COLOR[opt], alpha=0.15)

    final = {opt: np.array([r["test_acc_curve"][-1] for r in runs[opt]]).mean() * 100
             for opt in ("SGD", "Adam")}
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    arch_label = "MLP" if arch == "MLP" else "SmallCNN"
    ax.set_title(f"{arch_label} — Test Accuracy\n"
                 f"SGD: {final['SGD']:.2f}%  |  Adam: {final['Adam']:.2f}%")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    save(fig, f"fig_test_acc_{tag}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 & 4  — Train-test gap curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_gap(arch, runs, tag):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    epochs = np.arange(1, len(runs["SGD"][0]["test_acc_curve"]) + 1)

    for opt in ("SGD", "Adam"):
        gaps = [
            (np.array(r["train_acc_curve"]) - np.array(r["test_acc_curve"]))
            for r in runs[opt]
        ]
        mu, sd = mean_std(gaps)
        mu_pct, sd_pct = mu * 100, sd * 100
        ax.plot(epochs, mu_pct, color=COLOR[opt], ls=LS[opt],
                lw=1.8, label=opt)
        ax.fill_between(epochs, mu_pct - sd_pct, mu_pct + sd_pct,
                        color=COLOR[opt], alpha=0.15)

    final_gap = {
        opt: np.array([
            (r["train_acc_curve"][-1] - r["test_acc_curve"][-1])
            for r in runs[opt]
        ]).mean() * 100
        for opt in ("SGD", "Adam")
    }
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train − Test accuracy (%)")
    arch_label = "MLP" if arch == "MLP" else "SmallCNN"
    ax.set_title(f"{arch_label} — Train–Test Gap\n"
                 f"SGD: {final_gap['SGD']:.2f}%  |  Adam: {final_gap['Adam']:.2f}%")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    save(fig, f"fig_gap_{tag}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 & 6  — Sharpness vs gap scatter
# ─────────────────────────────────────────────────────────────────────────────
def plot_sharpness_gap(arch, runs, tag):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    all_sharpness, all_gap = [], []

    for opt in ("SGD", "Adam"):
        sh  = np.array([r["final_sharpness"] for r in runs[opt]])
        gap = np.array([
            (r["train_acc_curve"][-1] - r["test_acc_curve"][-1]) * 100
            for r in runs[opt]
        ])
        ax.scatter(sh, gap, color=COLOR[opt], marker="o", s=60,
                   label=opt, zorder=3, edgecolors="white", linewidths=0.5)
        # cluster centroid
        ax.scatter(sh.mean(), gap.mean(), color=COLOR[opt], marker="D",
                   s=120, zorder=4, edgecolors="black", linewidths=0.8)
        all_sharpness.extend(sh.tolist())
        all_gap.extend(gap.tolist())

    # Pearson r across all 10 points
    r = np.corrcoef(all_sharpness, all_gap)[0, 1]

    ax.set_xlabel(r"$\lambda_{\max}(H)$")
    ax.set_ylabel("Train − Test gap (%)")
    arch_label = "MLP" if arch == "MLP" else "SmallCNN"
    ax.set_title(f"{arch_label} — Sharpness vs. Generalization Gap\n"
                 f"Pearson $r = {r:.2f}$ (all 10 points)")
    ax.legend(frameon=False, title="Optimizer")
    # annotate note
    note = "Circles = individual seeds\nDiamonds = cluster means"
    ax.text(0.97, 0.04, note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray")
    fig.tight_layout()
    save(fig, f"fig_sharpness_gap_{tag}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for arch, tag in [("MLP", "mlp"), ("SmallCNN", "cnn")]:
        print(f"\n--- {arch} ---")
        runs = load_runs(arch)
        plot_test_acc(arch, runs, tag)
        plot_gap(arch, runs, tag)
        plot_sharpness_gap(arch, runs, tag)

    print(f"\nAll figures saved to {OUT_DIR}/")
