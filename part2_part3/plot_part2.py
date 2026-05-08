"""
Generate Part II figures referenced in the report:
  param_distance.png    - L2 distance from initialization
  sharpness_trace.png   - lambda_max(H) and tr(H) bar comparison
  inter_seed.png        - inter-seed pairwise parameter distance

Outputs go to ../report/figures/part2_geometry/ so the LaTeX build picks them up.
"""
import os
import json
import itertools

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "./results_part2"
FIGURES_DIR = "../report/figures/part2_geometry"
SEEDS       = [42, 123, 456, 789, 2024]
EPOCHS      = 30

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


def load_run(arch, opt, seed):
    with open(os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{seed}.json")) as f:
        return json.load(f)


def collect_metric(arch, opt, key):
    return np.array([load_run(arch, opt, s)[key] for s in SEEDS])


def collect_scalar(arch, opt, key):
    return np.array([load_run(arch, opt, s)[key] for s in SEEDS])


def plot_mean_std(ax, data, color, label, alpha_fill=0.15):
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    ax.plot(EPOCH_AXIS, mean, color=color, label=label, linewidth=1.8)
    ax.fill_between(EPOCH_AXIS, mean - std, mean + std,
                    color=color, alpha=alpha_fill)


def figure_param_distance():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, arch in zip(axes, ["MLP", "SmallCNN"]):
        sgd_dist  = collect_metric(arch, "SGD",  "param_distances")
        adam_dist = collect_metric(arch, "Adam", "param_distances")
        plot_mean_std(ax, sgd_dist,  SGD_COLOR,  "SGD")
        plot_mean_std(ax, adam_dist, ADAM_COLOR, "Adam")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\|\theta_t - \theta_0\|_2$")
        ax.set_title(f"({'a' if arch == 'MLP' else 'b'}) {arch}")
        ax.legend()
    fig.suptitle("L2 Distance from Initialization", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "param_distance.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def figure_sharpness_trace():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric_key, ylabel, title in zip(
        axes,
        ["final_sharpness", "final_trace"],
        [r"Sharpness $\lambda_{\max}(H)$", r"Hessian Trace $\mathrm{tr}(H)$"],
        ["Sharpness Proxy", "Hessian Trace Proxy"],
    ):
        x_positions = np.array([0, 1.4])
        width = 0.5
        for i, (arch, x) in enumerate(zip(["MLP", "SmallCNN"], x_positions)):
            sgd_vals  = collect_scalar(arch, "SGD",  metric_key)
            adam_vals = collect_scalar(arch, "Adam", metric_key)
            ax.bar(x - width/2, sgd_vals.mean(), width, yerr=sgd_vals.std(),
                   color=SGD_COLOR,  capsize=4,
                   label="SGD"  if i == 0 else "_nolegend_")
            ax.bar(x + width/2, adam_vals.mean(), width, yerr=adam_vals.std(),
                   color=ADAM_COLOR, capsize=4,
                   label="Adam" if i == 0 else "_nolegend_")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["MLP", "SmallCNN"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Hessian-Based Curvature Proxies at Final Epoch",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "sharpness_trace.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def figure_inter_seed():
    from models import MLP, SmallCNN
    arch_cls = {"MLP": MLP, "SmallCNN": SmallCNN}

    def load_flat(arch, opt, s):
        npy = os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{s}_params.npy")
        if os.path.exists(npy):
            return torch.tensor(np.load(npy))
        pt = os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{s}_state.pt")
        model = arch_cls[arch]()
        model.load_state_dict(torch.load(pt, map_location="cpu"))
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    def pairwise_dists(arch, opt):
        vecs = [load_flat(arch, opt, s) for s in SEEDS]
        return [
            (vecs[i] - vecs[j]).norm().item()
            for i, j in itertools.combinations(range(len(vecs)), 2)
        ]

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
        bp["boxes"][0].set_facecolor(SGD_COLOR  + "99")
        bp["boxes"][1].set_facecolor(ADAM_COLOR + "99")
        bp["boxes"][0].set_edgecolor(SGD_COLOR)
        bp["boxes"][1].set_edgecolor(ADAM_COLOR)
        for k, (vals, color) in enumerate(
            [(sgd_pw, SGD_COLOR), (adam_pw, ADAM_COLOR)], start=1
        ):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), k) + jitter, vals,
                       color=color, s=20, alpha=0.7, zorder=3)
        ax.set_ylabel(r"Pairwise $\|\theta^*_i - \theta^*_j\|_2$")
        ax.set_title(f"({'a' if arch == 'MLP' else 'b'}) {arch}")
    fig.suptitle("Inter-Seed Parameter Dispersion", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "inter_seed.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figure_param_distance()
    figure_sharpness_trace()
    figure_inter_seed()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
