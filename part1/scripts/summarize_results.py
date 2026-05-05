"""Aggregate results across experiments and generate combined summary + all plots.

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
from src.plotting import generate_all_plots
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True,
                        help='One or more experiment YAML config paths')
    args = parser.parse_args()

    all_summaries = []

    for config_path in args.configs:
        cfg = load_config(config_path)
        base_dir = Path(cfg['output']['base_dir'])
        exp_name = cfg['experiment']['name']

        print(f"\n=== Aggregating: {exp_name} ===")
        try:
            agg_results = build_aggregated_outputs(cfg, base_dir)
            print(f"  Saved: {base_dir}/aggregated_*.csv, report_ready_summary.md")

            generate_all_plots(cfg, agg_results, base_dir)
            all_summaries.append((exp_name, base_dir))
        except FileNotFoundError as e:
            print(f"  WARNING: Missing data for {exp_name} — run run_multi_seed.py first.\n  {e}")
            continue

    # Combined summary
    if len(all_summaries) > 0:
        output_path = Path('outputs') / 'report_ready_summary.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Combined Experiment Report\n\n")
            f.write("This file aggregates results from all experiments.\n\n")
            for exp_name, base_dir in all_summaries:
                sub_summary = (base_dir / 'report_ready_summary.md')
                if sub_summary.exists():
                    f.write(f"\n---\n\n")
                    f.write(sub_summary.read_text(encoding='utf-8'))
        print(f"\nCombined summary saved to {output_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
