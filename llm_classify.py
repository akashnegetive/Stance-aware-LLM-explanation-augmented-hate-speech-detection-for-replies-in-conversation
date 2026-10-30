"""LLM-only baseline: the LLM labels each reply as hate / neutral / counter directly (no fine-tuning).

Answers the reviewer question "does the LLM solve the task by itself?". Same gold test set and metrics
as the fine-tuned models; writes results.json + test_predictions.jsonl in the same format, so
significance.py can compare it with the trained models.

  --shots 0   zero-shot: label definitions only
  --shots K   few-shot: K fixed examples per class from the gold TRAIN split, shown as earlier chat turns

Resumable: answers are appended to raw.jsonl after every batch and skipped on restart.

Example (Kaggle, 2 x T4):
  python llm_classify.py --shots 0 --output_dir /kaggle/working/outputs
  python llm_classify.py --shots 2 --output_dir /kaggle/working/outputs
"""

import argparse
import json
import os
import re
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from generate_llm import generate, load_llm, load_pairs

LABELS = {"hate": 0, "neutral": 1, "counter": 2}  # same ids as the dataset (see baseline.LABEL_NAMES)

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


def pick_shots(data_dir, k, max_chars=300):
    """k short training examples per class, in file order (deterministic)."""
    shots = {lab: [] for lab in LABELS}
    with open(os.path.join(data_dir, "gold", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lab = {v: n for n, v in LABELS.items()}[int(r["label"])]
            if len(shots[lab]) < k and len(r["context"]) + len(r["target"]) <= max_chars:
                shots[lab].append(r)
    # interleave classes so the order gives no hint
    return [shots[lab][i] for i in range(k) for lab in LABELS if i < len(shots[lab])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--shots", type=int, default=0, help="examples per class from the training split (0 = zero-shot)")
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--max_context_chars", type=int, default=1500)
    p.add_argument("--limit", type=int, default=0, help="only the first N test pairs (smoke test)")
    args = p.parse_args()

    name = f"llm_{args.model.split('/')[-1]}_{args.shots}shot" + (f"_lim{args.limit}" if args.limit else "")
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw.jsonl")

    gold = {}
    with open(os.path.join(args.data_dir, "gold", "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold[r["idx"]] = int(r["label"])
    pairs = load_pairs(args.data_dir, ["test"], args.limit)

    done = {}
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    done[o["idx"]] = o
    todo = [x for x in pairs if x["idx"] not in done]
    print(f"[{name}] {len(pairs)} test pairs, {len(done)} already done, {len(todo)} to classify")

    def user_turn(context, target):
        return {"role": "user", "content": PROMPT.format(context=context[:args.max_context_chars], target=target)}

    history = [{"role": "system", "content": SYSTEM}]
    for s in pick_shots(args.data_dir, args.shots) if args.shots else []:
        lab = {v: n for n, v in LABELS.items()}[int(s["label"])]
        history += [user_turn(s["context"], s["target"]), {"role": "assistant", "content": json.dumps({"label": lab})}]

    if todo:
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
                    o = {"idx": x["idx"], "pred_label": label, "parse_ok": ok, "raw": text.strip()}
                    done[x["idx"]] = o
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
                f.flush()
                n = b + len(batch)
                print(f"[{name}] {n}/{len(todo)} done | {n / (time.time() - start):.2f} pairs/s", flush=True)

    idxs = [x["idx"] for x in pairs]
    y_true = np.array([gold[i] for i in idxs])
    y_pred = np.array([LABELS[done[i]["pred_label"]] for i in idxs])
    names = list(LABELS)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=names, digits=4,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=names, digits=4, zero_division=0))
    parse_rate = float(np.mean([done[i]["parse_ok"] for i in idxs]))
    results = {
        "run_name": name,
        "args": vars(args),
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
    print(f"[{name}] weighted F1 {100 * t['test_weighted_f1']:.2f} | macro F1 {100 * t['test_macro_f1']:.2f} | "
          f"parsed {100 * parse_rate:.1f}% -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
