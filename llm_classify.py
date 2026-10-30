"""LLM-only baseline: the LLM labels each reply as hate / neutral / counter directly (no fine-tuning).

Answers the reviewer question "does the LLM solve the task by itself?". Same gold test set and metrics
as the fine-tuned models; writes results.json + test_predictions.jsonl in the same format, so
significance.py can compare it with the trained models.

  --shots 0             zero-shot: label definitions only
  --shots K             few-shot: K examples per class from the gold TRAIN split, shown as earlier chat turns
  --shot_seeds 0 1 2    which example sets: 0 = the first short examples in file order, N > 0 = a random draw
                        with seed N. Several seeds run with one model load, to measure few-shot variance.
  --splits train val test
                        also label train/val pairs, so the labels can be used as model input (baseline.py
                        --use_llm_label). Metrics are computed on test only.

Resumable: answers are appended to <run>/raw.jsonl after every batch and skipped on restart
(--resume_from copies an earlier raw.jsonl in first).

Example (Kaggle, 2 x T4):
  python llm_classify.py --shots 0 --output_dir /kaggle/working/outputs
  python llm_classify.py --shots 2 --shot_seeds 1 2 3 --output_dir /kaggle/working/outputs
"""

import argparse
import json
import os
import random
import re
import shutil
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from generate_llm import generate, load_llm, load_pairs

LABELS = {"hate": 0, "neutral": 1, "counter": 2}  # same ids as the dataset (see baseline.LABEL_NAMES)
ID2LABEL = {v: k for k, v in LABELS.items()}

SYSTEM = "You are an expert annotator of online conversations. You answer only with JSON."

PROMPT = """Classify the REPLY, reading it in the context of the PARENT comment it responds to.

Labels:
- "hate": the reply is hateful, abusive or offensive toward a person or a group.
- "counter": the reply pushes back against hateful, abusive or offensive content (counter-speech).
- "neutral": neither of the above.

Parent comment:
\"\"\"{context}\"\"\"

Reply:
\"\"\"{target}\"\"\"

Return a JSON object: {{"label": "hate" | "neutral" | "counter"}}"""


def parse_label(text):
    """Return (label, parse_ok). Falls back to the first label word, then to "neutral"."""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            label = str(json.loads(m.group(0)).get("label", "")).strip().lower()
            if label in LABELS:
                return label, True
        except json.JSONDecodeError:
            pass
    w = re.search(r"\b(hate|neutral|counter)", text, re.I)
    return (w.group(1).lower(), False) if w else ("neutral", False)


def pick_shots(data_dir, k, seed=0, max_chars=300):
    """k short training examples per class. seed 0: the first ones in file order; seed N: a random draw."""
    pools = {lab: [] for lab in LABELS}
    with open(os.path.join(data_dir, "gold", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["context"]) + len(r["target"]) <= max_chars:
                pools[ID2LABEL[int(r["label"])]].append(r)
    rng = random.Random(seed)
    shots = {lab: (pool[:k] if seed == 0 else rng.sample(pool, k)) for lab, pool in pools.items()}
    # interleave classes so the order gives no hint
    return [shots[lab][i] for i in range(k) for lab in LABELS if i < len(shots[lab])]


def run_name(args, seed):
    name = f"llm_{args.model.split('/')[-1]}_{args.shots}shot"
    if args.shots and seed:
        name += f"_seed{seed}"
    return name + (f"_lim{args.limit}" if args.limit else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--shots", type=int, default=0, help="examples per class from the training split (0 = zero-shot)")
    p.add_argument("--shot_seeds", nargs="+", type=int, default=[0], help="example sets to try (0 = file order)")
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--resume_from", nargs="*", default=[],
                   help="earlier raw.jsonl files; each run continues from the most complete file in a folder of its name")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--max_context_chars", type=int, default=1500)
    p.add_argument("--limit", type=int, default=0, help="only the first N pairs of each split (smoke test)")
    args = p.parse_args()

    gold = {}
    for split in args.splits:
        with open(os.path.join(args.data_dir, "gold", f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                gold[(split, r["idx"])] = int(r["label"])
    pairs = load_pairs(args.data_dir, args.splits, args.limit)

    def user_turn(context, target):
        return {"role": "user", "content": PROMPT.format(context=context[:args.max_context_chars], target=target)}

    tok = model = None
    for seed in args.shot_seeds if args.shots else [0]:
        name = run_name(args, seed)
        out_dir = os.path.join(args.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        raw_path = os.path.join(out_dir, "raw.jsonl")
        if not os.path.exists(raw_path):
            # Earlier files of this run (same folder name); the most complete one wins.
            candidates = [f for f in args.resume_from
                          if os.path.isfile(f) and os.path.basename(os.path.dirname(f)) == name]
            if candidates:
                best = max(candidates, key=lambda f: sum(1 for _ in open(f, encoding="utf-8")))
                shutil.copyfile(best, raw_path)
                print(f"[{name}] continuing from {best}")

        done = {}
        if os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        o = json.loads(line)
                        done[(o.get("split", "test"), o["idx"])] = o  # early files had test only, without "split"
        todo = [x for x in pairs if (x["split"], x["idx"]) not in done]
        print(f"[{name}] {len(pairs)} pairs ({'/'.join(args.splits)}), {len(pairs) - len(todo)} already done, "
              f"{len(todo)} to classify")

        history = [{"role": "system", "content": SYSTEM}]
        for s in pick_shots(args.data_dir, args.shots, seed) if args.shots else []:
            history += [user_turn(s["context"], s["target"]),
                        {"role": "assistant", "content": json.dumps({"label": ID2LABEL[int(s["label"])]})}]

        if todo:
            if model is None:
                tok, model = load_llm(args.model)
            todo.sort(key=lambda x: len(x["context"]) + len(x["target"]))
            start = time.time()
            with open(raw_path, "a", encoding="utf-8") as f:
                for b in range(0, len(todo), args.batch_size):
                    batch = todo[b:b + args.batch_size]
                    texts = generate(tok, model, [history + [user_turn(x["context"], x["target"])] for x in batch],
                                     args.max_new_tokens)
                    for x, text in zip(batch, texts):
                        label, ok = parse_label(text)
                        o = {"split": x["split"], "idx": x["idx"], "pred_label": label, "parse_ok": ok, "raw": text.strip()}
                        done[(x["split"], x["idx"])] = o
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                    f.flush()
                    n = b + len(batch)
                    rate = n / (time.time() - start)
                    print(f"[{name}] {n}/{len(todo)} done | {rate:.2f} pairs/s | "
                          f"~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)

        if "test" not in args.splits:
            continue
        test = [x for x in pairs if x["split"] == "test"]
        idxs = [x["idx"] for x in test]
        y_true = np.array([gold[("test", i)] for i in idxs])
        y_pred = np.array([LABELS[done[("test", i)]["pred_label"]] for i in idxs])
        names = list(LABELS)
        report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=names, digits=4,
                                       output_dict=True, zero_division=0)
        parse_rate = float(np.mean([done[("test", i)]["parse_ok"] for i in idxs]))
        results = {
            "run_name": name,
            "args": {**vars(args), "shot_seed": seed},
            "system": "llm_baseline",
            "parse_rate": parse_rate,
            "test": {"test_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                     "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
                     "test_accuracy": float((y_true == y_pred).mean())},
            "test_report": report,
            "test_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(out_dir, "test_predictions.jsonl"), "w") as f:
            for i, yt, yp in zip(idxs, y_true, y_pred):
                f.write(json.dumps({"idx": i, "label": int(yt), "pred": int(yp)}) + "\n")
        t = results["test"]
        print(f"[{name}] test weighted F1 {100 * t['test_weighted_f1']:.2f} | macro F1 {100 * t['test_macro_f1']:.2f} | "
              f"parsed {100 * parse_rate:.1f}% -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
