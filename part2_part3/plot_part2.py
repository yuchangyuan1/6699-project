"""
Part II: Generate all figures and tables for the paper.

Figures produced (matching the paper):
  Figure 4 : L2 distance from initialization (MLP + CNN)
  Figure 5 : Average gradient norm per epoch (MLP + CNN)
  Figure 6 : Sharpness and Hessian trace bar comparisons (MLP + CNN)
  Figure 7 : Inter-seed parameter dispersion box plots (MLP + CNN)

Tables produced (printed to stdout and saved as .tex):
  Table 3  : Final scalar metrics (test acc, gap, sharpness, trace, param dist)
  Table 4  : Inter-seed parameter dispersion

Usage:
    python plot_part2.py
"""
import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "./results_part2"
FIGURES_DIR = "./figures_part2"
SEEDS       = [42, 123, 456, 789, 2024]
EPOCHS      = 30

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

SGD_COLOR  = "#2166ac"
ADAM_COLOR = "#d6604d"
EPOCH_AXIS = list(range(1, EPOCHS + 1))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_run(arch, opt, seed):
    """Load JSON result file for one run."""
    path = os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{seed}.json")
    with open(path) as f:
        return json.load(f)


def load_flat_params(arch, opt, seed):
    """Load saved flat parameter vector."""
    path = os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{seed}_params.npy")
    return torch.tensor(np.load(path))


def collect_metric(arch, opt, metric_key):
    """
    Return array of shape (n_seeds, n_epochs) for a per-epoch metric,
    or array of shape (n_seeds,) for a scalar metric.
    """
    rows = []
    for seed in SEEDS:
        run = load_run(arch, opt, seed)
        rows.append(run[metric_key])
    return np.array(rows)


def collect_scalar(arch, opt, scalar_key):
    vals = []
    for seed in SEEDS:
        run = load_run(arch, opt, seed)
        vals.append(run[scalar_key])
    return np.array(vals)


# ── Figure helpers ────────────────────────────────────────────────────────────

def plot_mean_std(ax, data, color, label, alpha_fill=0.15):
    """Plot mean ± std band for data of shape (n_seeds, n_epochs)."""
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    ax.plot(EPOCH_AXIS, mean, color=color, label=label, linewidth=1.8)
    ax.fill_between(EPOCH_AXIS, mean - std, mean + std,
                    color=color, alpha=alpha_fill)


# ── Figure 4: L2 distance from initialization ─────────────────────────────

def figure4_param_distance():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, arch in zip(axes, ["MLP", "SmallCNN"]):
        sgd_dist  = collect_metric(arch, "SGD",  "param_distances")
        adam_dist = collect_metric(arch, "Adam", "param_distances")

        plot_mean_std(ax, sgd_dist,  SGD_COLOR,  "SGD")
        plot_mean_std(ax, adam_dist, ADAM_COLOR, "Adam")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\|\theta_t - \theta_0\|_2$")
        ax.set_title(f"({['a','b'][list(['MLP','SmallCNN']).index(arch)]}) {arch}")
        ax.legend()

    fig.suptitle("L2 Distance from Initialization", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig4_param_distance.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 5: Gradient norm per epoch ────────────────────────────────────────

def figure5_gradient_norm():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, arch in zip(axes, ["MLP", "SmallCNN"]):
        sgd_gn  = collect_metric(arch, "SGD",  "grad_norms")
        adam_gn = collect_metric(arch, "Adam", "grad_norms")

        plot_mean_std(ax, sgd_gn,  SGD_COLOR,  "SGD")
        plot_mean_std(ax, adam_gn, ADAM_COLOR, "Adam")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Average $\|g_t\|_2$")
        ax.set_title(f"({['a','b'][list(['MLP','SmallCNN']).index(arch)]}) {arch}")
        ax.legend()

    fig.suptitle("Average Gradient Norm per Epoch", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig5_gradient_norm.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 6: Sharpness and Hessian trace ────────────────────────────────────

def figure6_sharpness_trace():
    """
    Two-column figure:
      Left : sharpness (lambda_max) bar comparison, MLP vs CNN
      Right: Hessian trace bar comparison, MLP vs CNN
    Each column has two grouped bars (SGD vs Adam) per architecture.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric_key, ylabel, title_suffix in zip(
        axes,
        ["final_sharpness", "final_trace"],
        [r"Sharpness $\lambda_{\max}(H)$", r"Hessian Trace $\mathrm{tr}(H)$"],
        ["Sharpness Proxy", "Hessian Trace Proxy"],
    ):
        x_positions = np.array([0, 1.4])   # two architecture groups
        width = 0.5

        for i, (arch, x) in enumerate(zip(["MLP", "SmallCNN"], x_positions)):
            sgd_vals  = collect_scalar(arch, "SGD",  metric_key)
            adam_vals = collect_scalar(arch, "Adam", metric_key)

            ax.bar(x - width/2, sgd_vals.mean(),  width, yerr=sgd_vals.std(),
                   color=SGD_COLOR,  capsize=4,
                   label="SGD"  if i == 0 else "_nolegend_")
            ax.bar(x + width/2, adam_vals.mean(), width, yerr=adam_vals.std(),
                   color=ADAM_COLOR, capsize=4,
                   label="Adam" if i == 0 else "_nolegend_")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(["MLP", "SmallCNN"])
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)
        ax.legend()

    fig.suptitle("Hessian-Based Curvature Proxies at Final Epoch",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig6_sharpness_trace.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 7: Inter-seed parameter dispersion ─────────────────────────────────

def figure7_inter_seed_dispersion():
    """
    For each (arch, optimizer), compute pairwise distances between
    final parameter vectors across seeds and show as box plots.
    """
    import itertools, torch

    def pairwise_dists(arch, opt):
        vecs = [torch.tensor(np.load(
            os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{s}_params.npy")
        )) for s in SEEDS]
        dists = []
        for i, j in itertools.combinations(range(len(vecs)), 2):
            dists.append((vecs[i] - vecs[j]).norm().item())
        return dists

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, arch in zip(axes, ["MLP", "SmallCNN"]):
        sgd_pw  = pairwise_dists(arch, "SGD")
        adam_pw = pairwise_dists(arch, "Adam")

        bp = ax.boxplot(
            [sgd_pw, adam_pw],
            labels=["SGD", "Adam"],
            patch_artist=True,
            widths=0.4,
            medianprops={"color": "black", "linewidth": 2},
        )
        bp["boxes"][0].set_facecolor(SGD_COLOR  + "99")  # semi-transparent
        bp["boxes"][1].set_facecolor(ADAM_COLOR + "99")
        bp["boxes"][0].set_edgecolor(SGD_COLOR)
        bp["boxes"][1].set_edgecolor(ADAM_COLOR)

        # Overlay individual points
        for k, (vals, color) in enumerate(
            [(sgd_pw, SGD_COLOR), (adam_pw, ADAM_COLOR)], start=1
        ):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), k) + jitter, vals,
                       color=color, s=20, alpha=0.7, zorder=3)

        ax.set_ylabel(r"Pairwise $\|\theta^*_i - \theta^*_j\|_2$")
        ax.set_title(f"({['a','b'][list(['MLP','SmallCNN']).index(arch)]}) {arch}")

    fig.suptitle("Inter-Seed Parameter Dispersion", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig7_inter_seed.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Tables ────────────────────────────────────────────────────────────────────

def table3_final_metrics():
    """Print Table 3: final scalar metrics."""
    rows = []
    for arch in ["MLP", "SmallCNN"]:
        for opt in ["SGD", "Adam"]:
            test_accs  = collect_scalar(arch, opt, "test_acc_curve")[:, -1] \
                         if False else \
                         np.array([load_run(arch, opt, s)["test_acc_curve"][-1]
                                   for s in SEEDS])
            train_accs = np.array([load_run(arch, opt, s)["train_acc_curve"][-1]
                                   for s in SEEDS])
            gaps       = train_accs - test_accs
            sharp      = collect_scalar(arch, opt, "final_sharpness")
            trace      = collect_scalar(arch, opt, "final_trace")
            dists      = np.array([load_run(arch, opt, s)["param_distances"][-1]
                                   for s in SEEDS])
            rows.append({
                "Model": arch, "Opt": opt,
                "Test acc": f"{test_accs.mean():.4f}+/-{test_accs.std():.4f}",
                "Gap":      f"{gaps.mean():.4f}+/-{gaps.std():.4f}",
                "Sharpness":f"{sharp.mean():.2f}+/-{sharp.std():.2f}",
                "Trace":    f"{trace.mean():.2f}+/-{trace.std():.2f}",
                "Param dist":f"{dists.mean():.2f}+/-{dists.std():.2f}",
            })

    header = ["Model","Opt","Test acc","Gap","Sharpness","Trace","Param dist"]
    col_w  = [max(len(h), max(len(r[h]) for r in rows)) for h in header]
    fmt    = "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n=== Table 3: Final Scalar Metrics ===")
    print(fmt.format(*header))
    print("-" * (sum(col_w) + 2*(len(col_w)-1)))
    for r in rows:
        print(fmt.format(*[r[h] for h in header]))

    # Save LaTeX version
    tex_rows = []
    for r in rows:
        cells = [r[h] for h in header]
        cells_tex = [c.replace("+/-", r" $\pm$ ") for c in cells]
        tex_rows.append(" & ".join(cells_tex) + r" \\")

    tex = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Final scalar metrics across five seeds. Gap = train acc $-$ test acc.}" + "\n"
        r"\label{tab:part2-scalar}" + "\n"
        r"\small" + "\n"
        r"\begin{tabular}{llccccc}" + "\n"
        r"\toprule" + "\n"
        " & ".join(header) + r" \\" + "\n"
        r"\midrule" + "\n"
    )
    tex += "\n".join(tex_rows) + "\n"
    tex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"

    out = os.path.join(FIGURES_DIR, "table3_final_metrics.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX: {out}")


def table4_interseed_dispersion():
    """Print Table 4: inter-seed parameter dispersion."""
    import itertools, torch

    print("\n=== Table 4: Inter-Seed Parameter Dispersion ===")
    rows = []
    for arch in ["MLP", "SmallCNN"]:
        for opt in ["SGD", "Adam"]:
            vecs = [torch.tensor(np.load(
                os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{s}_params.npy")
            )) for s in SEEDS]
            dists = []
            for i, j in itertools.combinations(range(len(vecs)), 2):
                dists.append((vecs[i] - vecs[j]).norm().item())
            mean_dist = np.mean(dists)
            rows.append((arch, opt, mean_dist))
            print(f"  {arch:10s} {opt:5s}  mean pairwise dist = {mean_dist:.2f}")

    # Save LaTeX
    tex = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Inter-seed parameter dispersion: mean pairwise L2 distance "
        r"between final parameter vectors across five seeds.}" + "\n"
        r"\label{tab:part2-dispersion}" + "\n"
        r"\begin{tabular}{llc}" + "\n"
        r"\toprule" + "\n"
        r"Architecture & Optimizer & Mean pairwise dist \\" + "\n"
        r"\midrule" + "\n"
    )
    for arch, opt, d in rows:
        tex += f"{arch} & {opt} & {d:.2f}" + r" \\" + "\n"
    tex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"

    out = os.path.join(FIGURES_DIR, "table4_dispersion.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Generating Figure 4: L2 distance from initialization...")
    figure4_param_distance()

    print("Generating Figure 5: Gradient norm per epoch...")
    figure5_gradient_norm()

    print("Generating Figure 6: Sharpness and Hessian trace...")
    figure6_sharpness_trace()

    print("Generating Figure 7: Inter-seed parameter dispersion...")
    figure7_inter_seed_dispersion()

    print("Generating Table 3: Final scalar metrics...")
    table3_final_metrics()

    print("Generating Table 4: Inter-seed dispersion...")
    table4_interseed_dispersion()

    print(f"\nAll outputs in: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
