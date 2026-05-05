# Adam vs SGD Convergence Experiment

Reproducible PyTorch framework comparing Adam and SGD convergence speed on Fashion-MNIST.

**Research question**: Does Adam converge faster than SGD in early-stage training?

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Execution Order

### Step 1: LR Search (find best learning rates)
```bash
python scripts/run_lr_search.py --config configs/experiment_c_lr_search.yaml
```
Check `outputs/exp_c_lr_search/lr_search_summary.csv` for best LRs.

### Step 2: MLP Main Experiment (Experiment A)
```bash
python scripts/run_multi_seed.py --config configs/experiment_a_mlp.yaml
```

### Step 3: Small CNN Experiment (Experiment B)
```bash
python scripts/run_multi_seed.py --config configs/experiment_b_cnn.yaml
```

### Step 4: Generate Combined Summary & All Plots
```bash
python scripts/summarize_results.py \
    --configs configs/experiment_a_mlp.yaml configs/experiment_b_cnn.yaml
```

---

## Single Run (for debugging)
```bash
python scripts/run_single.py --config configs/experiment_a_mlp.yaml --optimizer adam --seed 42
```

---

## Output Structure

```
outputs/
├── exp_a_mlp/
│   ├── sgd/seed_42/{metrics.csv, step_metrics.csv, thresholds.csv, summary.json}
│   ├── adam/seed_42/{...}
│   ├── aggregated_metrics.csv        # epoch-level mean±std across seeds
│   ├── aggregated_step_metrics.csv   # step-level mean±std
│   ├── aggregated_thresholds.csv     # threshold crossing stats
│   ├── report_ready_summary.md       # tabular summary
│   └── plots/
│       ├── loss_vs_epoch.{png,pdf}
│       ├── loss_vs_step.{png,pdf}
│       ├── early_stage_zoom.{png,pdf}
│       └── threshold_comparison.{png,pdf}
├── exp_b_cnn/  (same structure)
├── exp_c_lr_search/
│   ├── lr_search_summary.csv
│   └── plots/lr_search.{png,pdf}
└── report_ready_summary.md           # combined report
```

---

## Reproducibility

- All experiments use `torch.manual_seed` + `numpy.random.seed` + `random.seed`
- `torch.backends.cudnn.deterministic = True`
- DataLoader uses a seeded `torch.Generator` for shuffle
- Both SGD and Adam runs for the same seed start from **identical model weights** (via `reset_parameters` called after seeding)
- GPU results may show minor floating-point differences across platforms; CPU results are fully deterministic

---

## Fair Comparison Design

| Shared | Changed |
|--------|---------|
| Dataset (Fashion-MNIST) | Optimizer type (SGD vs Adam) |
| Model architecture | Optimizer hyperparameters |
| Weight initialization | |
| Batch size (128) | |
| Epochs (30) | |
| Loss function (CrossEntropyLoss) | |
| Random seed | |

**SGD**: lr=0.01, momentum=0.9  
**Adam**: lr=0.001, betas=(0.9, 0.999)

---

## Models

**MLP**: Flatten → Linear(784,256) → BN → ReLU → Linear(256,128) → BN → ReLU → Linear(128,10)

**SmallCNN**: Conv(1,16) → BN → ReLU → Pool → Conv(16,32) → BN → ReLU → Pool → Linear(1568,128) → ReLU → Linear(128,10)
