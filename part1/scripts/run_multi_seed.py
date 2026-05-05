"""Run all seeds × optimizers for a given config.

Usage:
    python scripts/run_multi_seed.py --config configs/experiment_a_mlp.yaml
    python scripts/run_multi_seed.py --config configs/experiment_a_mlp.yaml --optimizers sgd adam
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import build_aggregated_outputs
from src.train import run_experiment
from src.utils import get_output_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--optimizers', nargs='+', default=['sgd', 'adam'])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg['experiment']['seeds']
    base_dir = cfg['output']['base_dir']
    exp_name = cfg['experiment']['name']

    total = len(args.optimizers) * len(seeds)
    done = 0
    all_summaries = []

    for opt_name in args.optimizers:
        for seed in seeds:
            done += 1
            print(f"\n[{done}/{total}] {exp_name} | {opt_name} | seed={seed}")
            output_dir = get_output_dir(base_dir, opt_name, seed)
            summary = run_experiment(cfg, opt_name, seed, output_dir)
            all_summaries.append(summary)
            print(f"  Final loss: {summary['final_epoch_loss']:.4f}")

    print(f"\nAll runs complete. Building aggregated outputs in {base_dir}/")
    build_aggregated_outputs(cfg, Path(base_dir))
    print("Done.")


if __name__ == '__main__':
    main()
