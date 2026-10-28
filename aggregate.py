"""Collect outputs/*/results.json into a mean +- std table over seeds.

Usage: python aggregate.py --output_dir outputs
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.output_dir, "*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        setting = re.sub(r"_s\d+$", "", r["run_name"])
        rep = r["test_report"]
        groups[setting].append({
            "macro_f1": r["test"]["test_macro_f1"],
            "weighted_f1": r["test"]["test_weighted_f1"],
            "accuracy": r["test"]["test_accuracy"],
            "train_min": (r.get("train_runtime_sec") or 0) / 60,
            **{f"f1_{k}": v["f1-score"] for k, v in rep.items() if isinstance(v, dict) and k not in
               ("macro avg", "weighted avg")},
        })

    if not groups:
        print(f"No results found in {args.output_dir}/*/results.json")
        return

    cols = list(next(iter(groups.values()))[0].keys())
    header = "| setting | n | " + " | ".join(cols) + " |"
    print(header)
    print("|" + "---|" * (len(cols) + 2))
    for setting, runs in groups.items():
        cells = []
        for c in cols:
            # Scores are shown as percentages; train_min is already in minutes.
            v = np.array([r[c] for r in runs]) * (1 if c == "train_min" else 100)
            cells.append(f"{v.mean():.2f} ± {v.std():.2f}")
        print(f"| {setting} | {len(runs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
