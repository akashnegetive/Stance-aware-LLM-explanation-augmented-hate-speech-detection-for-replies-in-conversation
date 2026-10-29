"""Learning-rate tuning for the baselines, selected on the gold *validation* set only.

Stage 1 (search):  every learning rate x input setting, on a few seeds.
Stage 2 (final):   the best learning rate per input setting (highest mean validation score), on all seeds.
The test set is never used for choosing; it is only reported for the final configuration.
Every run goes through baseline.py, so finished runs are cached and never retrained.

Example (about 70 min for roberta-base on a T4):
  python tune.py --model roberta-base --select_metric weighted_f1
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from baseline import build_parser, make_run_name


def run(cfg, seed, args):
    run_args = ["--model", args.model, "--input", cfg["input"], "--lr", str(cfg["lr"]),
                "--select_metric", args.select_metric, "--seed", str(seed),
                "--data_dir", args.data_dir, "--output_dir", args.output_dir]
    if args.prev_results:
        run_args += ["--prev_results", *args.prev_results]
    # Each run is a separate process so GPU memory is fully released between runs.
    subprocess.run([sys.executable, "baseline.py", *run_args], check=True)
    parser = build_parser()
    run_name = make_run_name(parser.parse_args(run_args), vars(parser.parse_args([])))
    with open(os.path.join(args.output_dir, run_name, "results.json")) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--inputs", nargs="+", default=["target", "context_target"])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-5, 2e-5, 3e-5])
    p.add_argument("--select_metric", choices=["macro_f1", "weighted_f1"], default="weighted_f1")
    p.add_argument("--search_seeds", nargs="+", type=int, default=[13, 42])
    p.add_argument("--final_seeds", nargs="+", type=int, default=[13, 42, 87, 123, 2024])
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--prev_results", nargs="*", default=[])
    args = p.parse_args()

    val_key = f"val_{args.select_metric}"
    lines = [f"# Tuning: {args.model}, selected on validation {args.select_metric}", ""]

    for inp in args.inputs:
        # Stage 1: search
        scores = {}
        for lr in args.lrs:
            cfg = {"input": inp, "lr": lr}
            vals = [run(cfg, s, args)["val"][val_key] for s in args.search_seeds]
            scores[lr] = float(np.mean(vals))
        best_lr = max(scores, key=scores.get)

        lines += [f"## {inp}", "", "| lr | mean val " + args.select_metric + " (search seeds) |", "|---|---|"]
        lines += [f"| {lr:g}{' **(best)**' if lr == best_lr else ''} | {100 * v:.2f} |" for lr, v in scores.items()]

        # Stage 2: best lr on all seeds
        final = [run({"input": inp, "lr": best_lr}, s, args) for s in args.final_seeds]
        test_w = np.array([r["test"]["test_weighted_f1"] for r in final]) * 100
        test_m = np.array([r["test"]["test_macro_f1"] for r in final]) * 100
        lines += ["", f"Final (lr={best_lr:g}, {len(final)} seeds): test weighted F1 = {test_w.mean():.2f} ± {test_w.std():.2f}, "
                      f"test macro F1 = {test_m.mean():.2f} ± {test_m.std():.2f}", ""]

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.output_dir, f"tuning_{args.model.replace('/', '_')}_{args.select_metric}.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
