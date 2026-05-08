# Part I - Early-Stage Convergence

Code for Section "Part I: Early-Stage Convergence" of the report. Trains an
MLP and a SmallCNN on Fashion-MNIST with SGD and Adam over five matched seeds
and records the first global step at which each run crosses fixed
training-loss thresholds.

## Layout

```
configs/   experiment_{a_mlp,b_cnn,c_lr_search}.yaml
scripts/   run_single.py | run_multi_seed.py | run_lr_search.py | summarize_results.py
src/       train.py | analysis.py | metrics.py | models.py | datasets.py | utils.py
outputs/   per-seed CSVs and aggregated CSVs (exp_a_mlp, exp_b_cnn, exp_c_lr_search)
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Optional learning-rate sweep on the MLP
python scripts/run_lr_search.py --config configs/experiment_c_lr_search.yaml

# 2. Five-seed MLP and CNN runs
python scripts/run_multi_seed.py --config configs/experiment_a_mlp.yaml
python scripts/run_multi_seed.py --config configs/experiment_b_cnn.yaml

# 3. Aggregate to combined CSVs and a summary markdown
python scripts/summarize_results.py \
    --configs configs/experiment_a_mlp.yaml configs/experiment_b_cnn.yaml
```

Single-run debugging:

```bash
python scripts/run_single.py --config configs/experiment_a_mlp.yaml --optimizer adam --seed 42
```

## Outputs

Per-seed run (`outputs/exp_*/<opt>/seed_<n>/`):

* `metrics.csv`        - per-epoch mean loss
* `step_metrics.csv`   - per-step loss
* `thresholds.csv`     - first step at each loss threshold
* `summary.json`       - run metadata

Aggregated (`outputs/exp_*/`):

* `aggregated_metrics.csv`        - per-epoch mean / std across seeds
* `aggregated_step_metrics.csv`   - per-step mean / std (smoothed)
* `aggregated_thresholds.csv`     - threshold-crossing means and stds (Table I in the report)
* `report_ready_summary.md`       - human-readable summary

Fashion-MNIST is downloaded into `data/` automatically on first run.
