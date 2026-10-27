"""Run every method on one dataset and print its row of Table 4.

    python run_benchmark.py --dataname adult --num_samples 1000

It first prepares what the methods need -- the shared black box, and TABCF's VAE --
reusing existing checkpoints, then dispatches each method through main.py as a
subprocess (sample, then evaluate), so all of them see the same black box and the
same factual instances. A method that fails is logged and does not stop the rest.
At the end every results.csv is collected into
counterfactual_results/<dataname>/summary_<n>_samples.csv.
"""

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

from evaluate import METHOD_FOLDERS, results_dir

# Ours first, then the baselines in the order of Table 4.
DEFAULT_ORDER = ["flowcf", "wachter", "dice", "revise", "cchvae", "tabcf", "dicoflex"]

COLUMNS = ["Validity ↑", "Sparsity Cat ↓", "Sparsity Num ↓", "Prox Num ↓"]


def run_step(method, mode, args, log_path):
    cmd = [sys.executable, "main.py",
           "--dataname", args.dataname,
           "--method", method,
           "--mode", mode,
           "--num_samples", str(args.num_samples),
           "--gpu", str(args.gpu)]
    start = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode == 0, time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Run all CF methods on one dataset")
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Factual instances to explain (-1 = all).")
    parser.add_argument("--methods", type=str, default=None,
                        help=f"Comma-separated subset. Default: {','.join(DEFAULT_ORDER)}")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "sample", "evaluate"],
                        help="'evaluate' re-scores existing counterfactuals.")
    parser.add_argument("--save_path", type=str, default="counterfactual_results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else list(DEFAULT_ORDER)
    unknown = [m for m in methods if m not in METHOD_FOLDERS]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}. Choose from: {', '.join(METHOD_FOLDERS)}")

    log_dir = os.path.join(args.log_dir, args.dataname)
    os.makedirs(log_dir, exist_ok=True)
    print(f"dataset={args.dataname}  n={args.num_samples}  methods: {', '.join(methods)}")
    print(f"logs: {log_dir}/<method>_{args.num_samples}.log\n")

    status = {}
    if args.mode in ("all", "sample"):
        # every method explains the same black box; TABCF additionally needs its VAE.
        # Both steps return immediately once the checkpoint exists.
        prep = [("blackbox", "train")] + ([("vae", "train")] if "tabcf" in methods else [])
        for method, mode in prep:
            print(f"[prep] {method:9s} ", end="", flush=True)
            log_path = os.path.join(log_dir, f"{method}_train.log")
            open(log_path, "w").close()
            ok, secs = run_step(method, mode, args, log_path)
            print(f"{'ok' if ok else 'FAILED'}  {secs / 60:.1f} min"
                  + ("" if ok else f" -> {log_path}"))
            if not ok:
                sys.exit(1)

    for i, method in enumerate(methods, 1):
        log_path = os.path.join(log_dir, f"{method}_{args.num_samples}.log")
        open(log_path, "w").close()
        print(f"[{i}/{len(methods)}] {method:9s} ", end="", flush=True)

        elapsed, ok = 0.0, True
        for mode in (["sample", "evaluate"] if args.mode == "all" else [args.mode]):
            step_ok, secs = run_step(method, mode, args, log_path)
            elapsed += secs
            if not step_ok:
                print(f"FAILED at {mode} after {elapsed / 60:.1f} min -> {log_path}")
                status[method] = f"failed ({mode})"
                ok = False
                break
        if ok:
            print(f"ok  {elapsed / 60:.1f} min")
            status[method] = "ok"

    rows = []
    for method in methods:
        csv = os.path.join(results_dir(args.save_path, args.dataname, method, args.num_samples),
                           "results.csv")
        if os.path.exists(csv):
            row = pd.read_csv(csv, index_col=0).iloc[0].to_dict()
            row["method"] = method
            rows.append(row)

    print()
    if not rows:
        print("No results.csv found — check the logs.")
        return

    summary = pd.DataFrame(rows).set_index("method")[COLUMNS]
    pd.set_option("display.width", 200)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    out = os.path.join(args.save_path, args.dataname, f"summary_{args.num_samples}_samples.csv")
    summary.to_csv(out)
    print(f"\nsaved: {out}")

    failed = {m: s for m, s in status.items() if s != "ok"}
    if failed:
        print("\nFailed: " + ", ".join(f"{m} [{s}]" for m, s in failed.items()))
        sys.exit(1)


if __name__ == "__main__":
    main()
