"""Run a single (model, optimizer, seed) Part I training run.

Usage:
    python scripts/run_single.py --config configs/experiment_a_mlp.yaml --optimizer sgd --seed 42
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import run_experiment
from src.utils import get_output_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--optimizer', required=True, choices=['sgd', 'adam'])
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = get_output_dir(cfg['output']['base_dir'], args.optimizer, args.seed)

    print(f"Running: {cfg['experiment']['name']} | {args.optimizer} | seed={args.seed}")
    summary = run_experiment(cfg, args.optimizer, args.seed, output_dir)

    print(f"Done. Final loss: {summary['final_epoch_loss']:.4f} | "
          f"Time: {summary['training_time_sec']}s | Output: {output_dir}")
    for t, info in summary['thresholds'].items():
        if info:
            print(f"  loss <= {t}: epoch={info['epoch']}, global_step={info['global_step']}")
        else:
            print(f"  loss <= {t}: NOT REACHED")


if __name__ == '__main__':
    main()
