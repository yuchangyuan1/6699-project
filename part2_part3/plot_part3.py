"""
Generate Part III sharpness-vs-gap scatter figures referenced in the report:
  sharpness_gap_mlp.png
  sharpness_gap_cnn.png

Outputs go to ../report/figures/part3_generalization/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS   = [42, 123, 456, 789, 2024]
RESULTS = "results_part2"
OUT_DIR = "../report/figures/part3_generalization"

COLOR = {"SGD": "#2166ac", "Adam": "#d6604d"}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def load_runs(arch):
    return {
        opt: [json.load(open(os.path.join(RESULTS, f"{arch}_{opt}_seed{s}.json")))
              for s in SEEDS]
        for opt in ("SGD", "Adam")
    }


def plot_sharpness_gap(arch, runs, tag):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    all_sharp, all_gap = [], []
    for opt in ("SGD", "Adam"):
        sh  = np.array([r["final_sharpness"] for r in runs[opt]])
        gap = np.array([
            (r["train_acc_curve"][-1] - r["test_acc_curve"][-1]) * 100
            for r in runs[opt]
        ])
        ax.scatter(sh, gap, color=COLOR[opt], marker="o", s=60,
                   label=opt, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(sh.mean(), gap.mean(), color=COLOR[opt], marker="D",
                   s=120, zorder=4, edgecolors="black", linewidths=0.8)
        all_sharp.extend(sh.tolist())
        all_gap.extend(gap.tolist())

    r = np.corrcoef(all_sharp, all_gap)[0, 1]
    ax.set_xlabel(r"$\lambda_{\max}(H)$")
    ax.set_ylabel("Train - Test gap (%)")
    arch_label = "MLP" if arch == "MLP" else "SmallCNN"
    ax.set_title(f"{arch_label} - Sharpness vs. Generalization Gap\n"
                 f"Pearson $r = {r:.2f}$ (all 10 points)")
    ax.legend(frameon=False, title="Optimizer")
    ax.text(0.97, 0.04, "Circles = individual seeds\nDiamonds = cluster means",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"sharpness_gap_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for arch, tag in [("MLP", "mlp"), ("SmallCNN", "cnn")]:
        plot_sharpness_gap(arch, load_runs(arch), tag)
    print(f"\nAll figures saved to {OUT_DIR}/")
