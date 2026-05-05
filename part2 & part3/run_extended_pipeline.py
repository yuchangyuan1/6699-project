"""
Run the full Plan B evaluation pipeline once train_part2.py has finished.

Order:
    1. perturbation_flatness.py
    2. mode_connectivity.py
    3. function_space.py
    4. representation_cka.py
    5. loss_matched_geometry.py
    6. plot_extended.py

Each step writes its own JSON / figures and is idempotent. Use --skip to skip
already-completed steps by JSON existence (useful when iterating).

Usage:
    python run_extended_pipeline.py
    python run_extended_pipeline.py --skip
"""
import os
import sys
import time
import argparse
import subprocess


STEPS = [
    ("perturbation_flatness.py",  "results_part2/perturbation_flatness.json"),
    ("mode_connectivity.py",      "results_part2/mode_connectivity.json"),
    ("function_space.py",         "results_part2/function_space.json"),
    ("representation_cka.py",     "results_part2/representation_cka.json"),
    ("loss_matched_geometry.py",  "results_part2/loss_matched_geometry.json"),
    ("plot_extended.py",          None),  # always re-runs (cheap)
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", action="store_true",
                   help="Skip steps whose output JSON already exists")
    p.add_argument("--only", default=None,
                   help="Run only one script (e.g. 'mode_connectivity.py')")
    return p.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for script, out_json in STEPS:
        if args.only and args.only != script:
            continue
        if args.skip and out_json and os.path.exists(out_json):
            print(f"[skip] {script} (output exists: {out_json})")
            continue
        print(f"\n{'='*62}")
        print(f"  RUNNING: {script}")
        print(f"{'='*62}", flush=True)
        rc = subprocess.run([sys.executable, script], env=env).returncode
        if rc != 0:
            print(f"[fail] {script} exited with {rc}", flush=True)
            sys.exit(rc)

    print(f"\nPipeline done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
