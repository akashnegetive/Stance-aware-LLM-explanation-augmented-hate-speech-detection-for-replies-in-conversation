"""Phase 3-4 ablation: base / +LLM explanation / +stance / +LLM+stance, for every encoder.

Every run goes through baseline.py, so finished runs are cached and an interrupted session
continues where it stopped. Seeds are the outer loop, so a partial run still covers all variants.

Example:
  python ablation.py --llm_file llm/llm_outputs.jsonl \
      --models roberta-base=2e-5 microsoft/deberta-v3-base=2e-5 --select_metric weighted_f1
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from baseline import build_parser, make_run_name

VARIANTS = {
    "base": [],
    "llm": ["--use_explanation"],
    "stance": ["--stance_weight", "{w}"],
    "llm_stance": ["--use_explanation", "--stance_weight", "{w}"],
}


def run_args(model, lr, variant, seed, args):
    a = ["--model", model, "--lr", str(lr), "--input", "context_target", "--seed", str(seed),
         "--select_metric", args.select_metric, "--data_dir", args.data_dir, "--output_dir", args.output_dir]
    a += [x.format(w=args.stance_weight) for x in VARIANTS[variant]]
    if variant != "base":
        a += ["--llm_file", args.llm_file]
    if args.limit:
        a += ["--limit", str(args.limit), "--epochs", "1"]
    if args.prev_results:
        a += ["--prev_results", *args.prev_results]
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["roberta-base=2e-5"],
                   help="model=learning_rate pairs (use the learning rate chosen in tuning)")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87, 123, 2024])
    p.add_argument("--stance_weight", type=float, default=0.5)
    p.add_argument("--select_metric", choices=["macro_f1", "weighted_f1"], default="weighted_f1")
    p.add_argument("--llm_file", default="llm/llm_outputs.jsonl")
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--prev_results", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0, help="smoke test: N pairs per split, 1 epoch")
    args = p.parse_args()

    models = [(m.split("=")[0], float(m.split("=")[1]) if "=" in m else 2e-5) for m in args.models]
    parser = build_parser()
    defaults = vars(parser.parse_args([]))

    results = {}
    for model, lr in models:
        for seed in args.seeds:
            for variant in args.variants:
                a = run_args(model, lr, variant, seed, args)
                subprocess.run([sys.executable, "baseline.py", *a], check=True)
                name = make_run_name(parser.parse_args(a), defaults)
                with open(os.path.join(args.output_dir, name, "results.json")) as f:
                    results.setdefault((model, variant), []).append(json.load(f))

    lines = [f"# Ablation (selected on validation {args.select_metric}, stance weight {args.stance_weight:g}, "
             f"{len(args.seeds)} seeds)", "",
             "| model | variant | weighted F1 | Δ vs base | macro F1 | Δ vs base | hate F1 | neutral F1 | counter F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model, lr in models:
        base = results.get((model, "base"))
        for variant in args.variants:
            runs = results[(model, variant)]
            w = np.array([r["test"]["test_weighted_f1"] for r in runs]) * 100
            m = np.array([r["test"]["test_macro_f1"] for r in runs]) * 100
            per = {c: np.mean([r["test_report"][c]["f1-score"] for r in runs]) * 100 for c in ("hate", "neutral", "counter")}
            if base and variant != "base":
                bw = np.mean([r["test"]["test_weighted_f1"] for r in base]) * 100
                bm = np.mean([r["test"]["test_macro_f1"] for r in base]) * 100
                dw, dm = f"{w.mean() - bw:+.2f}", f"{m.mean() - bm:+.2f}"
            else:
                dw = dm = "–"
            lines.append(f"| {model} (lr {lr:g}) | {variant} | {w.mean():.2f} ± {w.std():.2f} | {dw} | "
                         f"{m.mean():.2f} ± {m.std():.2f} | {dm} | {per['hate']:.1f} | {per['neutral']:.1f} | {per['counter']:.1f} |")

    report = "\n".join(lines)
    print(report)
    suffix = "_smoke" if args.limit else ""
    with open(os.path.join(args.output_dir, f"ablation_{args.select_metric}{suffix}.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
