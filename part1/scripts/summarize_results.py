"""Aggregate per-seed results across one or more experiment configs.

Usage:
    python scripts/summarize_results.py \
        --configs configs/experiment_a_mlp.yaml configs/experiment_b_cnn.yaml
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import build_aggregated_outputs
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True,
                        help='One or more experiment YAML config paths')
    args = parser.parse_args()

    summaries = []
    for config_path in args.configs:
        cfg = load_config(config_path)
        base_dir = Path(cfg['output']['base_dir'])
        exp_name = cfg['experiment']['name']
        print(f"\n=== Aggregating: {exp_name} ===")
        try:
            build_aggregated_outputs(cfg, base_dir)
            print(f"  Saved: {base_dir}/aggregated_*.csv, report_ready_summary.md")
            summaries.append((exp_name, base_dir))
        except FileNotFoundError as e:
            print(f"  WARNING: missing data for {exp_name}; run run_multi_seed.py first.\n  {e}")

    if summaries:
        out = Path('outputs') / 'report_ready_summary.md'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write("# Combined Experiment Report\n\n")
            for exp_name, base_dir in summaries:
                sub = base_dir / 'report_ready_summary.md'
                if sub.exists():
                    f.write("\n---\n\n" + sub.read_text(encoding='utf-8'))
        print(f"\nCombined summary saved to {out}")


if __name__ == '__main__':
    main()
