# Part II + Part III - Solution Geometry and Generalization

Code and results for the "Part II: Solution Geometry" and
"Part III: Generalization and the Propagation of Implicit Bias" sections of
the report (`../report/revised_report.tex`).

## File map

```
part2_part3/
|-- data_utils.py              Fashion-MNIST loaders (matched-seed shuffle)
|-- models.py                  MLP / SmallCNN definitions
|-- model_io.py                checkpoint load helpers
|-- geometry.py                Hessian sharpness, trace, parameter distance
|
|-- train_part2.py             Train 20 runs (2 arch * 2 opt * 5 seeds);
|                              logs per-epoch path length, step norm, etc.
|
|-- perturbation_flatness.py   Operational flatness check (Sec. Perturbation Flatness)
|-- loss_matched_geometry.py   SGD-vs-Adam at matched training loss (Sec. Loss-Matched Control)
|-- mode_connectivity.py       Linear-interpolation barrier (Sec. Basin Geometry)
|-- function_space.py          D_pred / D_SKL / C_logit on the test set
|-- representation_cka.py      Feature-kernel alignment + linear CKA
|-- baseline_same_optimizer.py Cross-seed same-optimizer falsification baseline
|
|-- run_pipeline.py            Run every evaluation/plot script in order
|-- plot_part2.py              param_distance, sharpness_trace, inter_seed figures
|-- plot_part3.py              sharpness_gap_{mlp,cnn} figures
|-- plot_pipeline_figures.py   trajectory, perturbation, mode-connectivity, function-space figures
|-- dump_numbers.py            Print every number used in the report tables
|
`-- results_part2/             JSON results, final-epoch state_dicts, flat parameter vectors
```

## Reproduce

```bash
cd part2_part3

# 1. 20 training runs (2 archs * 2 opts * 5 seeds). Writes results_part2/.
python train_part2.py

# 2. All evaluation scripts + figure generation.
python run_pipeline.py

# 3. Print every number in the report tables.
python dump_numbers.py > numbers.txt
```

`run_pipeline.py --skip` skips analysis steps whose JSON output already exists;
`run_pipeline.py --only mode_connectivity.py` re-runs a single step.

## Result -> report mapping

| Report section                            | JSON                                                | Figure file (in `../report/figures/`)                                         |
|-------------------------------------------|-----------------------------------------------------|-------------------------------------------------------------------------------|
| II - Distance from Initialization         | `<arch>_<opt>_seed<n>.json`                         | `part2_geometry/param_distance.png`                                           |
| II - Trajectory Decomposition             | `<arch>_<opt>_seed<n>.json`                         | `part2_geometry/path_length_curve_{mlp,cnn}.png`, `step_norm_curve_*.png`     |
| II - Sharpness and Hessian Trace          | `<arch>_<opt>_seed<n>.json`                         | `part2_geometry/sharpness_trace.png`                                          |
| II - Perturbation Flatness                | `perturbation_flatness.json`                        | `part2_geometry/perturbation_curve_{mlp,cnn}.png`                             |
| II - Inter-Seed Dispersion                | `<arch>_<opt>_seed<n>_params.npy`                   | `part2_geometry/inter_seed.png`                                               |
| II - Loss-Matched Control                 | `loss_matched_geometry.json`                        | (table only)                                                                  |
| III - Test acc & Sharpness-Gap            | `<arch>_<opt>_seed<n>.json`                         | `part3_generalization/sharpness_gap_{mlp,cnn}.png`                            |
| III - Basin Geometry                      | `mode_connectivity.json`, `baseline_same_optimizer.json` | `part3_generalization/mode_connectivity_{mlp,cnn}.png`                  |
| III - Function-Space Similarity           | `function_space.json`, `baseline_same_optimizer.json`    | `part3_generalization/function_space_bars.png`                          |
| III - Representation-Space Alignment      | `representation_cka.json`, `baseline_same_optimizer.json`| (table only)                                                            |

## Notes

* `loss_matched_geometry.py` reads per-epoch checkpoints from
  `results_part2/checkpoints/<arch>_<opt>_seed<n>/epoch_NN.pt`. These are
  written by `train_part2.py` but not shipped with the submitted package
  to keep the bundle small. The final JSON it produces (`loss_matched_geometry.json`)
  is shipped, so you only need to retrain if you change training settings.
* `results_part2/` is referenced as a relative path inside several JSONs
  (`checkpoint_dir`), so renaming it will invalidate the saved paths.
* Fashion-MNIST is downloaded into `./data/` automatically on first run.
