"""
Run the full Part II/III evaluation pipeline (after train_part2.py).

Each step writes a JSON under results_part2/ and a figure under
../report/figures/. The pipeline is idempotent: --skip skips steps whose
JSON already exists; --only runs a single script by name.
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
    ("baseline_same_optimizer.py","results_part2/baseline_same_optimizer.json"),
    ("plot_part2.py",             None),
    ("plot_part3.py",             None),
    ("plot_pipeline_figures.py",  None),
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
        print(f"\n  RUNNING: {script}", flush=True)
        rc = subprocess.run([sys.executable, script], env=env).returncode
        if rc != 0:
            print(f"[fail] {script} exited with {rc}", flush=True)
            sys.exit(rc)

    print(f"\nPipeline done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
