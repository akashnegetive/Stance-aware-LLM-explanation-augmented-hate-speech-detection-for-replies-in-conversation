"""Significance tests and error analysis for the ablation (CPU only, a few minutes).

Reads every finished run (results.json + test_predictions.jsonl) under the given directories and, for
each encoder, compares every variant (+LLM, +stance, +LLM+stance) with that encoder's base run
(same learning rate, selection metric and seeds):

  1. paired t-test over seeds on test weighted and macro F1, with Holm correction across all comparisons;
  2. paired bootstrap over the 713 test pairs (resampling pairs; score = mean over seeds);
  3. error analysis of base vs +LLM using the seed ensemble (mean of softmax over seeds): which gold
     classes change, and example replies the explanation fixes or breaks, with the LLM's explanation.

LLM-only baselines from llm_classify.py found in the same directories are compared with each encoder's
+LLM variant by the paired bootstrap.

Example (Kaggle):
  python significance.py --results_dirs /kaggle/input/**/outputs --llm_file <llm_outputs.jsonl> \
      --out_dir /kaggle/working/outputs
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats

from baseline import LABEL_NAMES, NON_TRAINING_ARGS, build_parser


def variant_name(cfg):
    expl, w, lab = cfg["use_explanation"], cfg["stance_weight"], cfg.get("use_llm_label", False)
    parts = (["llm"] if expl else []) + (["stance"] if w else []) + (["label"] if lab else [])
    return ("_".join(parts) or "base") + (f" (w={w:g})" if w and w != 0.5 else "")


def load_runs(dirs, select_metric):
    defaults = vars(build_parser().parse_args([]))
    runs, llm_baselines = {}, {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*", "results.json")):
            with open(path) as f:
                r = json.load(f)
            pred_path = os.path.join(os.path.dirname(path), "test_predictions.jsonl")
            if not os.path.isfile(pred_path):
                continue
            if r.get("system") == "llm_baseline":
                if not r["args"].get("limit"):
                    llm_baselines.setdefault(r["run_name"], (r, pred_path))
                continue
            cfg = {**defaults, **r["args"]}
            if cfg["limit"] or cfg["input"] != "context_target" or cfg["select_metric"] != select_metric:
                continue
            runs.setdefault(r["run_name"], (cfg, r, pred_path))  # the same run may be attached twice
    return runs, llm_baselines


def read_preds(path):
    with open(path) as f:
        rows = sorted((json.loads(l) for l in f if l.strip()), key=lambda o: o["idx"])
    idx = np.array([o["idx"] for o in rows])
    y = np.array([o["label"] for o in rows])
    pred = np.array([o["pred"] for o in rows])
    logits = np.array([o["logits"] for o in rows]) if "logits" in rows[0] else None
    return idx, y, pred, logits


def f1_scores(y, p, k=3):
    """(weighted F1, macro F1) from a confusion matrix; fast enough for thousands of bootstrap samples."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float((f1 * support).sum() / support.sum()), float(f1.mean())


def bootstrap(y, preds_a, preds_b, n_boot, rng):
    """Paired bootstrap over test pairs. preds_* = list of prediction arrays (one per seed).
    Returns {metric: (mean diff a-b, 95% CI low, high, two-sided p)}."""
    n = len(y)
    diffs = np.empty((n_boot, 2))
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        a = np.mean([f1_scores(y[s], p[s]) for p in preds_a], axis=0)
        c = np.mean([f1_scores(y[s], p[s]) for p in preds_b], axis=0)
        diffs[b] = a - c
    full = np.mean([f1_scores(y, p) for p in preds_a], axis=0) - np.mean([f1_scores(y, p) for p in preds_b], axis=0)
    out = {}
    for j, m in enumerate(("weighted_f1", "macro_f1")):
        d = diffs[:, j]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[m] = (100 * full[j], 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5), max(p, 1 / n_boot))
    return out


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ensemble(preds):
    """Seed ensemble of fine-tuned runs: argmax of the mean softmax. Returns (predictions, members)."""
    return np.mean([softmax(p[3]) for p in preds], 0).argmax(1), len(preds)


def majority_vote(preds):
    """Majority vote over LLM draws (in name order, so the original draw comes first).
    Ties go to the tied label that appears in the earliest draw. Returns (predictions, members)."""
    P = np.stack([p[2] for p in preds])
    out = np.empty(P.shape[1], dtype=int)
    for i in range(P.shape[1]):
        counts = np.bincount(P[:, i], minlength=3)
        tied = set(np.flatnonzero(counts == counts.max()))
        out[i] = next(int(lab) for lab in P[:, i] if lab in tied)
    return out, len(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+", required=True, help="dirs holding <run>/results.json (globs allowed)")
    p.add_argument("--select_metric", default="weighted_f1")
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--llm_file", default=None, help="Phase 3a output, to show explanations in the error analysis")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--examples", type=int, default=5, help="examples per error-analysis category")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    dirs = sorted({d for pattern in args.results_dirs for d in glob.glob(pattern, recursive=True)})
    runs, llm_baselines = load_runs(dirs, args.select_metric)
    print(f"{len(runs)} ablation runs and {len(llm_baselines)} LLM baselines found in {len(dirs)} dirs")

    # group runs: (model, lr, other settings) -> variant -> seed -> run
    groups = {}
    for cfg, r, pred_path in runs.values():
        key_cfg = {k: v for k, v in cfg.items()
                   if k not in NON_TRAINING_ARGS | {"seed", "use_explanation", "stance_weight", "use_llm_label"}}
        key = json.dumps(key_cfg, sort_keys=True)
        var = variant_name(cfg)
        groups.setdefault(key, {}).setdefault(var, {})[cfg["seed"]] = (r, pred_path)

    rng = np.random.default_rng(0)
    rows, enc_llm, enc_vs_llm, enc_base = [], {}, {}, {}
    for key, variants in sorted(groups.items()):
        cfg = json.loads(key)
        model = f"{cfg['model']} (lr {cfg['lr']:g})" + (f" [{cfg['dataset']}]" if cfg["dataset"] != "yu2022" else "")
        if "base" not in variants:
            print(f"skipping {model}: no base runs")
            continue
        for var, seeds_v in sorted(variants.items()):
            if var == "base":
                continue
            seeds = sorted(set(seeds_v) & set(variants["base"]))
            if len(seeds) < 2:
                print(f"skipping {model} {var}: fewer than 2 seeds shared with base")
                continue
            res = {}
            for m in ("weighted_f1", "macro_f1"):
                a = np.array([seeds_v[s][0]["test"][f"test_{m}"] for s in seeds]) * 100
                b = np.array([variants["base"][s][0]["test"][f"test_{m}"] for s in seeds]) * 100
                t = stats.ttest_rel(a, b)
                res[m] = {"diff": (a - b).mean(), "p_t": float(t.pvalue), "wins": int((a > b).sum())}
            preds_v = [read_preds(seeds_v[s][1]) for s in seeds]
            preds_b = [read_preds(variants["base"][s][1]) for s in seeds]
            y = preds_v[0][1]
            boot = bootstrap(y, [x[2] for x in preds_v], [x[2] for x in preds_b], args.n_boot, rng)
            rows.append({"model": model, "variant": var, "n": len(seeds), **{m: {**res[m], "boot": boot[m]} for m in res}})
            if var == "llm":
                enc_llm[model] = (preds_b, preds_v)
            if var in ("llm", "label", "llm_label"):
                enc_vs_llm[(model, var)] = preds_v
            enc_base[model] = preds_b

    lines = [f"# Significance tests (test set, selected on validation {args.select_metric})", "",
             f"Paired t-test over seeds (Holm-corrected across the {len(rows)} comparisons) and paired bootstrap over "
             f"the test pairs ({args.n_boot} samples; Δ = variant − base, mean over seeds; 95% CI).", ""]
    for m in ("weighted_f1", "macro_f1"):
        adj = holm(np.array([r[m]["p_t"] for r in rows])) if rows else []
        lines += [f"## {m.replace('_', ' ')}", "",
                  "| model | variant | seeds | Δ | seeds better | t-test p | Holm p | bootstrap 95% CI | bootstrap p |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r, pa in zip(rows, adj):
            d, lo, hi, pb = r[m]["boot"]
            lines.append(f"| {r['model']} | {r['variant']} | {r['n']} | {r[m]['diff']:+.2f} | {r[m]['wins']}/{r['n']} | "
                         f"{r[m]['p_t']:.4f} | {pa:.4f} | [{lo:+.2f}, {hi:+.2f}] | {pb:.4f} |")
        lines.append("")

    # LLM-only baselines: draws of the same prompt with different example sets (seedN) are grouped
    if llm_baselines:
        configs = {}
        for name, (r, path) in sorted(llm_baselines.items()):
            configs.setdefault(re.sub(r"_seed\d+$", "", name), []).append((name, r, path))
        lines += ["## LLM-only baselines (mean ± std over example sets)", "",
                  "| LLM baseline | example sets | weighted F1 | macro F1 | per draw (weighted F1) | parsed |",
                  "|---|---|---|---|---|---|"]
        for cfg_name, draws in configs.items():
            w = np.array([d[1]["test"]["test_weighted_f1"] for d in draws]) * 100
            mf = np.array([d[1]["test"]["test_macro_f1"] for d in draws]) * 100
            parsed = np.mean([d[1].get("parse_rate", 1) for d in draws]) * 100
            lines.append(f"| {cfg_name} | {len(draws)} | {w.mean():.2f} ± {w.std():.2f} | {mf.mean():.2f} ± {mf.std():.2f} | "
                         f"{', '.join(f'{x:.1f}' for x in w)} | {parsed:.1f}% |")
        lines.append("")
        if enc_vs_llm:
            lines += ["## Fine-tuned models vs LLM-only baselines (paired bootstrap; LLM score = mean over example sets)", "",
                      "| fine-tuned | vs LLM baseline | Δ weighted F1 (95% CI, p) | Δ macro F1 (95% CI, p) |",
                      "|---|---|---|---|"]
            for (model, var), preds_v in sorted(enc_vs_llm.items()):
                for cfg_name, draws in configs.items():
                    preds_l = [read_preds(d[2]) for d in draws]
                    # same test pairs and gold labels, i.e. the same dataset
                    if not all(np.array_equal(p[0], preds_v[0][0]) and np.array_equal(p[1], preds_v[0][1]) for p in preds_l):
                        continue
                    b = bootstrap(preds_v[0][1], [x[2] for x in preds_v], [p[2] for p in preds_l], args.n_boot, rng)
                    cells = [f"{b[m][0]:+.2f} [{b[m][1]:+.2f}, {b[m][2]:+.2f}], p {b[m][3]:.4f}" for m in ("weighted_f1", "macro_f1")]
                    lines.append(f"| {model} +{var} | {cfg_name} | {cells[0]} | {cells[1]} |")
            lines.append("")

            # Ensemble vs ensemble: the fine-tuned seeds averaged (mean softmax) vs a majority vote of the LLM draws
            # Only systems scored on the same test pairs as the first fine-tuned model (one dataset per table).
            ref = next(iter(enc_vs_llm.values()))[0]
            same = lambda p: np.array_equal(p[0], ref[0]) and np.array_equal(p[1], ref[1])
            y = ref[1]
            ens = {f"{model} {'base' if var == 'base' else '+' + var}": ensemble(p)
                   for (model, var), p in sorted({**{(m, "base"): p for m, p in enc_base.items()}, **enc_vs_llm}.items())
                   if same(p[0])}
            votes = {}
            for cfg, draws in configs.items():
                preds_l = [read_preds(d[2]) for d in draws]
                if all(same(p) for p in preds_l):
                    votes[cfg] = majority_vote(preds_l)
            lines += ["## Ensembles: fine-tuned seeds (mean softmax) vs majority vote of the LLM's example sets", "",
                      "| system | members | weighted F1 | macro F1 |", "|---|---|---|---|"]
            for name, (pred, n) in ens.items():
                w, mf = f1_scores(y, pred)
                lines.append(f"| {name} | {n} seeds | {100 * w:.2f} | {100 * mf:.2f} |")
            for cfg, (pred, n) in votes.items():
                w, mf = f1_scores(y, pred)
                lines.append(f"| {cfg} (majority vote) | {n} example sets | {100 * w:.2f} | {100 * mf:.2f} |")
            lines += ["", "| fine-tuned ensemble | vs LLM vote | Δ weighted F1 (95% CI, p) | Δ macro F1 (95% CI, p) |",
                      "|---|---|---|---|"]
            for name, (pred, _) in ens.items():
                if name.endswith(" base"):
                    continue
                for cfg, (vote, _) in votes.items():
                    b = bootstrap(y, [pred], [vote], args.n_boot, rng)
                    cells = [f"{b[m][0]:+.2f} [{b[m][1]:+.2f}, {b[m][2]:+.2f}], p {b[m][3]:.4f}" for m in ("weighted_f1", "macro_f1")]
                    lines.append(f"| {name} | {cfg} | {cells[0]} | {cells[1]} |")
            lines += ["", "Cost per test pair: +llm and +label need one LLM call (explanation or label), +llm_label two, plus "
                          "one small encoder pass per seed; the LLM vote needs one LLM call per example set.", ""]

    # Error analysis: seed ensemble of base vs +LLM
    texts = {}
    with open(os.path.join(args.data_dir, "gold", "test.jsonl"), encoding="utf-8") as f:
        for l in f:
            o = json.loads(l)
            texts[o["idx"]] = o
    llm = {}
    if args.llm_file and os.path.isfile(args.llm_file):
        with open(args.llm_file, encoding="utf-8") as f:
            for l in f:
                o = json.loads(l)
                if o["split"] == "test":
                    llm[o["idx"]] = o
    N = LABEL_NAMES
    for model, (preds_b, preds_v) in enc_llm.items():
        idx, y = preds_b[0][0], preds_b[0][1]
        if not all(int(i) in texts and int(texts[int(i)]["label"]) == int(c) for i, c in zip(idx, y)):
            print(f"error analysis skipped for {model}: its test pairs are not the ones in {args.data_dir}")
            continue
        eb = np.mean([softmax(x[3]) for x in preds_b], 0).argmax(1)
        ev = np.mean([softmax(x[3]) for x in preds_v], 0).argmax(1)
        wb, mb = f1_scores(y, eb)
        wv, mv = f1_scores(y, ev)
        lines += [f"## Error analysis: {model}, base vs +LLM (ensemble of {len(preds_b)} seeds)", "",
                  f"Ensemble weighted / macro F1: base {100 * wb:.2f} / {100 * mb:.2f}, +LLM {100 * wv:.2f} / {100 * mv:.2f}.", "",
                  "| gold class | pairs | fixed by +LLM | broken by +LLM | net |", "|---|---|---|---|---|"]
        for c in sorted(N):
            sel = y == c
            fixed = int(((eb != y) & (ev == y) & sel).sum())
            broken = int(((eb == y) & (ev != y) & sel).sum())
            lines.append(f"| {N[c]} | {int(sel.sum())} | {fixed} | {broken} | {fixed - broken:+d} |")
        lines += ["", "Confusion (rows = gold, columns = predicted hate / neutral / counter):", "",
                  "| gold | base | +LLM |", "|---|---|---|"]
        for c in sorted(N):
            cb = np.bincount(eb[y == c], minlength=3)
            cv = np.bincount(ev[y == c], minlength=3)
            lines.append(f"| {N[c]} | {' / '.join(map(str, cb))} | {' / '.join(map(str, cv))} |")
        lines.append("")
        for title, mask in (("Fixed by +LLM", (eb != y) & (ev == y)), ("Broken by +LLM", (eb == y) & (ev != y))):
            lines += [f"### {title} (examples)", "",
                      "| gold | base → +LLM | parent (start) | reply | LLM stance: explanation |", "|---|---|---|---|---|"]
            for c in sorted(N):
                for i in np.where(mask & (y == c))[0][:args.examples]:
                    t = texts[int(idx[i])]
                    e = llm.get(int(idx[i]), {})
                    clean = lambda s, n: " ".join(s.split())[:n].replace("|", "/")
                    lines.append(f"| {N[c]} | {N[eb[i]]} → {N[ev[i]]} | {clean(t['context'], 160)} | {clean(t['target'], 160)} | "
                                 f"{e.get('stance', '')}: {clean(e.get('explanation', ''), 200)} |")
            lines.append("")

    report = "\n".join(lines)
    print(report)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"significance_{args.select_metric}.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
