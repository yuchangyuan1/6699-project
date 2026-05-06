# Plan B Pipeline (Extended Implicit-Bias Study)

## What's already done

| Step | File | Status |
|---|---|---|
| Modified training to log path length, step norm, per-epoch checkpoints | `train_part2.py` | done |
| Sanity-check single-seed fast run | — | passed (path length ≥ distance, BN buffers load correctly) |
| Full retrain with `--force` | `train_part2.py --force` | **in progress** |
| Perturbation flatness (§8) | `perturbation_flatness.py` | written, awaits retrain |
| Linear mode connectivity (§9) | `mode_connectivity.py` | written, awaits retrain |
| Function-space similarity (§10) | `function_space.py` | written, awaits retrain |
| Representation-space CKA (§11) | `representation_cka.py` | written, awaits retrain |
| Loss-matched geometry control (§7.1) | `loss_matched_geometry.py` | written, awaits retrain |
| Extended figures | `plot_extended.py` | written |
| LaTeX extension sections | `../6699 final project_v1 download from overleaf - newest/extension_sections.tex` | drafted, has `[PLACEHOLDER]` for numerical results |

## How to finish (after retrain completes)

```bash
cd "D:/yuchangyuan/Documents/6699 final project all parts/part2 & part3"

# Confirm all 20 v2 JSONs exist
python -c "import os, glob; n=len(glob.glob('results_part2/*_state.pt')); print(f'{n}/20 final state_dicts')"

# Run the full evaluation pipeline (~25-30 min on a 4060)
python run_extended_pipeline.py
```

This produces:
- `results_part2/perturbation_flatness.json`
- `results_part2/mode_connectivity.json`
- `results_part2/function_space.json`
- `results_part2/representation_cka.json`
- `results_part2/loss_matched_geometry.json`
- `figures_extended/*.png`

## Filling in LaTeX numbers

Open `extension_sections.tex` and replace each `[PLACEHOLDER]` / `TBD` with values from the JSON files. The mapping is:

| LaTeX section | JSON file | Field |
|---|---|---|
| §IV Trajectory Geometry | `results_part2/{arch}_{opt}_seed{s}.json` | `final_path_length`, `directness_ratio`, `mean_step_norm` (aggregate over 5 seeds) |
| §V Loss-Matched Control | `loss_matched_geometry.json` | `aggregated.{arch}.{sharp,trace,dist}_{S,A}_mean` |
| §VI Perturbation | `perturbation_flatness.json` | `aggregated.{arch}_{opt}.sigmas.{σ}.{mean,std}` (also embed the figure) |
| §VII Mode Connectivity | `mode_connectivity.json` | `aggregated.{arch}.barrier_{train,test}_{mean,std}` (embed figure) |
| §VIII Function-Space | `function_space.json` | `aggregated.{arch}.{D_pred,D_SKL,C_logit}_{mean,std}` |
| §IX Representation-Space | `representation_cka.json` | `aggregated.{arch}.{A,CKA}_{mean,std}` |

## Key paths

- v2 LaTeX (with `\input{extension_sections}`): `final_project_report_v2.tex` (next to original `_math_enhanced.tex`)
- Per-epoch SGD/Adam checkpoints (for §V loss-matched): `results_part2/checkpoints/{arch}_{opt}_seed{s}/epoch_{NN}.pt`
- Existing v1 figures (still referenced): `../6699 final project_v1 download from overleaf - newest/fig*.png`
- New figures: `figures_extended/*.png`

## Sanity checks built into the scripts

- `representation_cka.py` asserts CKA(K, K)=1 and CKA(K, αK)=1 on a synthetic feature matrix at start.
- `train_part2.py` logs `dist` and `P` per epoch — `P >= dist` must hold by triangle inequality.
- `mode_connectivity.py` evaluates λ=0 and λ=1 with original BN stats; the resulting losses should match the matched-seed end-of-training losses in the JSONs.
- `function_space.py` asserts SGD and Adam test loaders produce the same label order.

## Page-count sanity

Compile `final_project_report_v2.tex` after filling numbers. Target 22 pages, hard limit 25. If over 25, candidates to compress (in priority order):

1. Drop §IV path length figure or table (keep one, not both).
2. Move §VIII or §IX results table to a one-row summary in §X Three-Level Synthesis.
3. Move detailed loss-matched per-seed numbers to an appendix.
4. Tighten §1 Introduction to ~1 page.
