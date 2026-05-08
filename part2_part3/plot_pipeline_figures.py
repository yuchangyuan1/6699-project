"""
Generate the remaining Part II/III figures referenced in the report:

  Part II:
    path_length_curve_{mlp,cnn}.png   - cumulative path length vs distance
    step_norm_curve_{mlp,cnn}.png     - per-step update norm profile
    perturbation_curve_{mlp,cnn}.png  - perturbation flatness curve
  Part III:
    mode_connectivity_{mlp,cnn}.png   - linear interpolation loss curve
    function_space_bars.png           - D_pred / D_SKL / C_logit bars

Each figure is skipped if the corresponding result JSON is missing.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "./results_part2"
PART2_DIR   = "../report/figures/part2_geometry"
PART3_DIR   = "../report/figures/part3_generalization"
os.makedirs(PART2_DIR, exist_ok=True)
os.makedirs(PART3_DIR, exist_ok=True)

SEEDS      = [42, 123, 456, 789, 2024]
ARCHS      = ["MLP", "SmallCNN"]
SGD_COLOR  = "#2166ac"
ADAM_COLOR = "#d6604d"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _load_run(arch, opt, seed):
    p = os.path.join(RESULTS_DIR, f"{arch}_{opt}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _stack(arch, opt, key):
    rows = []
    for s in SEEDS:
        d = _load_run(arch, opt, s)
        if d is None or key not in d:
            return None
        rows.append(d[key])
    return np.array(rows)


def _line_with_band(ax, x, M, color, label):
    mu = M.mean(0)
    sd = M.std(0, ddof=0)
    ax.plot(x, mu, color=color, lw=2.0, label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2)


def plot_path_length():
    for arch in ARCHS:
        S   = _stack(arch, "SGD",  "path_length_per_epoch")
        A   = _stack(arch, "Adam", "path_length_per_epoch")
        D_S = _stack(arch, "SGD",  "param_distances")
        D_A = _stack(arch, "Adam", "param_distances")
        if any(x is None for x in (S, A, D_S, D_A)):
            print(f"  skip path_length ({arch}): missing fields")
            continue
        ep = np.arange(1, S.shape[1] + 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        ax = axes[0]
        _line_with_band(ax, ep, S, SGD_COLOR,  "SGD path length $\\mathcal{P}_T$")
        _line_with_band(ax, ep, A, ADAM_COLOR, "Adam path length $\\mathcal{P}_T$")
        _line_with_band(ax, ep, D_S, SGD_COLOR,  None)
        ax.lines[-1].set_linestyle("--"); ax.lines[-1].set_label(
            "SGD $\\|\\theta_t-\\theta_0\\|_2$")
        _line_with_band(ax, ep, D_A, ADAM_COLOR, None)
        ax.lines[-1].set_linestyle("--"); ax.lines[-1].set_label(
            "Adam $\\|\\theta_t-\\theta_0\\|_2$")
        ax.set_xlabel("Epoch"); ax.set_ylabel("L2 distance / path length")
        ax.set_title(f"{arch}: cumulative path length vs distance")
        ax.legend(loc="upper left", frameon=False)

        R_S = D_S / np.maximum(S, 1e-12)
        R_A = D_A / np.maximum(A, 1e-12)
        ax = axes[1]
        _line_with_band(ax, ep, R_S, SGD_COLOR,  "SGD")
        _line_with_band(ax, ep, R_A, ADAM_COLOR, "Adam")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Directness $R_t$")
        ax.set_title(f"{arch}: directness ratio $\\|\\theta_t-\\theta_0\\|/\\mathcal{{P}}_t$")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()
        suffix = "mlp" if arch == "MLP" else "cnn"
        out = os.path.join(PART2_DIR, f"path_length_curve_{suffix}.png")
        plt.savefig(out); plt.close(fig)
        print(f"  saved {out}")


def plot_step_norm():
    for arch in ARCHS:
        S = _stack(arch, "SGD",  "step_norm_per_epoch")
        A = _stack(arch, "Adam", "step_norm_per_epoch")
        if S is None or A is None:
            print(f"  skip step_norm ({arch}): missing fields")
            continue
        ep = np.arange(1, S.shape[1] + 1)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        _line_with_band(ax, ep, S, SGD_COLOR,  "SGD")
        _line_with_band(ax, ep, A, ADAM_COLOR, "Adam")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean $\\|\\Delta\\theta_t\\|_2$ in epoch")
        ax.set_title(f"{arch}: per-step update norm profile")
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()
        suffix = "mlp" if arch == "MLP" else "cnn"
        out = os.path.join(PART2_DIR, f"step_norm_curve_{suffix}.png")
        plt.savefig(out); plt.close(fig)
        print(f"  saved {out}")


def plot_perturbation():
    path = os.path.join(RESULTS_DIR, "perturbation_flatness.json")
    if not os.path.exists(path):
        print("  skip perturbation: result file missing")
        return
    with open(path) as f:
        d = json.load(f)
    sigmas = d["config"]["sigmas"]

    for arch in ARCHS:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for opt, color in [("SGD", SGD_COLOR), ("Adam", ADAM_COLOR)]:
            key = f"{arch}_{opt}"
            if key not in d.get("aggregated", {}):
                continue
            entry = d["aggregated"][key]
            mu = np.array([entry["sigmas"][f"{s:.0e}"]["mean"] for s in sigmas])
            sd = np.array([entry["sigmas"][f"{s:.0e}"]["std"]  for s in sigmas])
            ax.errorbar(sigmas, mu, yerr=sd, fmt="-o",
                        color=color, lw=2.0, capsize=3, label=opt)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("Perturbation scale $\\sigma$ (relative global)")
        ax.set_ylabel("$\\Delta\\hat{L}(\\sigma)$ on train batch")
        ax.set_title(f"{arch}: perturbation flatness")
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout()
        suffix = "mlp" if arch == "MLP" else "cnn"
        out = os.path.join(PART2_DIR, f"perturbation_curve_{suffix}.png")
        plt.savefig(out); plt.close(fig)
        print(f"  saved {out}")


def plot_mode_connectivity():
    path = os.path.join(RESULTS_DIR, "mode_connectivity.json")
    if not os.path.exists(path):
        print("  skip mode connectivity: result file missing")
        return
    with open(path) as f:
        d = json.load(f)
    lambdas = d["config"]["lambdas"]

    for arch in ARCHS:
        runs = [r for r in d["runs"] if r["arch"] == arch]
        if not runs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, key, ylabel in [
            (axes[0], "train_loss", "Train loss"),
            (axes[1], "test_loss",  "Test loss"),
        ]:
            curves = np.array([[c[key] for c in r["curves"]] for r in runs])
            mu = curves.mean(0); sd = curves.std(0, ddof=0)
            ax.plot(lambdas, mu, "-o", color="#444444", lw=2.0)
            ax.fill_between(lambdas, mu - sd, mu + sd, color="#888888", alpha=0.3)
            ax.set_xlabel("$\\lambda$  (0 = SGD, 1 = Adam)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{arch}: linear interpolation {ylabel.lower()}")
        plt.tight_layout()
        suffix = "mlp" if arch == "MLP" else "cnn"
        out = os.path.join(PART3_DIR, f"mode_connectivity_{suffix}.png")
        plt.savefig(out); plt.close(fig)
        print(f"  saved {out}")


def plot_function_space():
    path = os.path.join(RESULTS_DIR, "function_space.json")
    if not os.path.exists(path):
        print("  skip function-space: result file missing")
        return
    with open(path) as f:
        d = json.load(f)

    metrics = [("D_pred", "Prediction disagreement"),
               ("D_SKL",  "Symmetric KL"),
               ("C_logit","Logit cosine")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, label) in zip(axes, metrics):
        labels, means, stds = [], [], []
        for arch in ARCHS:
            agg = d["aggregated"].get(arch)
            if agg is None:
                continue
            labels.append(arch)
            means.append(agg[f"{key}_mean"])
            stds.append(agg[f"{key}_std"])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=["#888", "#444"], capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(label)
        ax.set_title(label)
    plt.tight_layout()
    out = os.path.join(PART3_DIR, "function_space_bars.png")
    plt.savefig(out); plt.close(fig)
    print(f"  saved {out}")


def main():
    print("Plotting pipeline figures...")
    plot_path_length()
    plot_step_norm()
    plot_perturbation()
    plot_mode_connectivity()
    plot_function_space()
    print(f"\nFigures saved under {PART2_DIR}/ and {PART3_DIR}/")


if __name__ == "__main__":
    main()
