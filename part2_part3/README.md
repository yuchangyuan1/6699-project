# Part II + Part III experiments

This directory contains every script and result that backs Sections "Part I:
Early-Stage Convergence", "Part II: Solution Geometry --- Multi-Faceted
Evidence", and "Part III: Generalization and the Propagation of Implicit
Bias" of the report.

The code is organized by which report section each file feeds.

## File map

```
part2_part3/
├── README.md                       # this file
│
├── data_utils.py                   # Fashion-MNIST loaders (shared)
├── models.py                       # MLP / SmallCNN definitions (shared)
├── model_io.py                     # checkpoint load/save (shared)
│
├── train_part2.py                  # Main training script.
│                                   # Logs path-length and step-norm trajectories
│                                   # used in Part II.
│
├── # ── Part II: Solution Geometry (multi-faceted evidence) ────────────
├── geometry.py                     # final-epoch sharpness / trace / distance /
│                                   # inter-seed dispersion (Sec. Distance,
│                                   # Sharpness, Inter-Seed Dispersion).
├── perturbation_flatness.py        # operational verification of curvature
│                                   # (Sec. Perturbation Flatness).
├── loss_matched_geometry.py        # confounder check: SGD vs Adam at matched
│                                   # training loss (Sec. Loss-Matched Control).
├── plot_part2.py                   # static Part II figures
│                                   # (param distance, sharpness/trace bars,
│                                   # inter-seed scatter).
│
├── # ── Part III: Generalization (geometry → predictions) ──────────────
├── baseline_same_optimizer.py      # cross-seed same-optimizer baseline used
│                                   # at every Part III level as a falsification
│                                   # reference.
├── mode_connectivity.py            # linear-interpolation barrier
│                                   # (Sec. Basin Geometry).
├── function_space.py               # D_pred / D_SKL / C_logit
│                                   # (Sec. Function-Space Similarity).
├── representation_cka.py           # feature-kernel alignment + linear CKA
│                                   # (Sec. Representation-Space Alignment).
├── plot_part3.py                   # static Part III figures
│                                   # (test acc / gap / sharpness-vs-gap).
│
├── # ── Pipeline & helpers ─────────────────────────────────────────────
├── run_pipeline.py                 # runs all Part II + Part III evidence
│                                   # scripts in dependency order.
├── plot_pipeline_figures.py        # generates per-architecture trajectory,
│                                   # perturbation, mode-connectivity and
│                                   # function/representation bar figures.
├── dump_numbers.py                 # prints every reported number with std
│                                   # for cross-checking the LaTeX tables.
│
└── results_part2/                  # all JSON / state.pt outputs land here
```

## How to reproduce

```bash
cd part2_part3

# 1. Train all 20 (arch × opt × seed) runs, with path-length / step-norm logging.
python train_part2.py --force

# 2. Sanity check
python -c "import glob; n=len(glob.glob('results_part2/*_state.pt')); print(f'{n}/20 final state_dicts')"

# 3. Run all Part II + Part III evidence experiments.
python run_pipeline.py             # ~25-30 min on a 4060

# 4. Generate static report figures.
python plot_part2.py
python plot_part3.py

# 5. Print the numbers used in the report tables.
python dump_numbers.py > numbers.txt
```

`run_pipeline.py --skip` skips steps whose JSON output already exists.

## Output → report mapping

The report tex file is `../report/final_project_report.tex`. Below is the
mapping from each result file to the report section that consumes it.

| Report section                                     | Result JSON                                  | Figures (under `report/figures/`)                                           |
|----------------------------------------------------|----------------------------------------------|-----------------------------------------------------------------------------|
| Part II / Distance from Initialization             | `*_seed*.json` (`final_distance`)            | `part2_geometry/param_distance.png`                                         |
| Part II / Trajectory Decomposition                 | `*_seed*.json` (`final_path_length`, `mean_step_norm`, `directness_ratio`) | `part2_geometry/path_length_curve_{mlp,cnn}.png`, `part2_geometry/step_norm_curve_{mlp,cnn}.png` |
| Part II / Sharpness and Hessian Trace              | `*_seed*.json` (`sharpness`, `trace`)        | `part2_geometry/sharpness_trace.png`                                        |
| Part II / Perturbation Flatness                    | `perturbation_flatness.json`                 | `part2_geometry/perturbation_curve_{mlp,cnn}.png`                           |
| Part II / Inter-Seed Dispersion                    | (computed from `*_seed*.json`)               | `part2_geometry/inter_seed.png`                                             |
| Part II / Loss-Matched Control                     | `loss_matched_geometry.json`                 | (table only)                                                                |
| Part III / Test acc & Sharpness--Gap               | `*_seed*.json`                               | `part3_generalization/sharpness_gap_{mlp,cnn}.png`                          |
| Part III / Basin Geometry (Mode Connectivity)      | `mode_connectivity.json`, `baseline_same_optimizer.json` | `part3_generalization/mode_connectivity_{mlp,cnn}.png`           |
| Part III / Function-Space Similarity               | `function_space.json`, `baseline_same_optimizer.json`    | `part3_generalization/function_space_bars.png`                   |
| Part III / Representation-Space Alignment          | `representation_cka.json`, `baseline_same_optimizer.json`| (table only)                                                     |

## Sanity checks built into the scripts

- `representation_cka.py` asserts CKA(K, K) = 1 and CKA(K, αK) = 1 on a
  synthetic feature matrix at start.
- `train_part2.py` logs `dist` and `P` per epoch — `P >= dist` must hold by
  the triangle inequality.
- `mode_connectivity.py` evaluates λ=0 and λ=1 with original BN stats; the
  resulting losses must match the matched-seed end-of-training losses in the
  per-seed JSONs.
- `function_space.py` asserts SGD and Adam test loaders produce the same
  label order.

## Result naming

`results_part2/` is the on-disk root of every experimental output and is used
verbatim as a relative path inside many JSON files (`checkpoint_dir`, etc.),
so it is intentionally kept as `results_part2/` even after the report
restructure. Renaming it would invalidate the saved checkpoint paths.
