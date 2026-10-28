"""Phase 1 baselines for context-aware hate / counter-speech detection on replies.

Reproduces the two baselines from Yu, Blanco & Hong (NAACL 2022):
  * target         -> classifier sees only the reply ("target")
  * context_target -> classifier sees parent comment ("context") + reply as a sentence pair

Data: https://github.com/xinchenyu/counter_context  (data/gold/{train,val,test}.jsonl)
Each line: {"idx": int, "label": "0"|"1"|"2", "context": str, "target": str}

Example:
  python baseline.py --input context_target --model roberta-base --seed 42
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Label ids as stored in the dataset. Names follow the paper's three classes;
# verify the id -> name mapping against the paper before reporting per-class numbers.
LABEL_NAMES = {0: "hate", 1: "neutral", 2: "counter"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r["label"] = int(r["label"])
        r["context"] = (r.get("context") or "").strip()
        r["target"] = (r.get("target") or "").strip()
    return rows


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, input_mode, max_len):
        if input_mode == "target":
            self.enc = tokenizer([r["target"] for r in rows], truncation=True, max_length=max_len)
        else:
            # Parent comment first, reply second. Long parents are truncated before the reply.
            self.enc = tokenizer(
                [r["context"] for r in rows],
                [r["target"] for r in rows],
                truncation="longest_first",
                max_length=max_len,
            )
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "accuracy": float((preds == labels).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--input", choices=["target", "context_target"], default="context_target")
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--use_silver", action="store_true", help="add silver train/val data to gold train/val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--keep_checkpoint", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    run_name = f"{args.model.replace('/', '_')}_{args.input}{'_silver' if args.use_silver else ''}_s{args.seed}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    train = read_jsonl(os.path.join(args.data_dir, "gold", "train.jsonl"))
    val = read_jsonl(os.path.join(args.data_dir, "gold", "val.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "gold", "test.jsonl"))
    if args.use_silver:
        # Silver data is only ever added to training; model selection and testing stay on gold.
        train += read_jsonl(os.path.join(args.data_dir, "silver", "train.jsonl"))
        train += read_jsonl(os.path.join(args.data_dir, "silver", "val.jsonl"))
    print(f"[{run_name}] train={len(train)} val={len(val)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_NAMES),
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    ds_train = PairDataset(train, tokenizer, args.input, args.max_len)
    ds_val = PairDataset(val, tokenizer, args.input, args.max_len)
    ds_test = PairDataset(test, tokenizer, args.input, args.max_len)

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "ckpt"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    val_metrics = trainer.evaluate(ds_val, metric_key_prefix="val")
    pred = trainer.predict(ds_test, metric_key_prefix="test")
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids
    names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    report = classification_report(y_true, y_pred, target_names=names, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    results = {
        "run_name": run_name,
        "args": vars(args),
        "val": val_metrics,
        "test": pred.metrics,
        "test_report": report,
        "test_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Per-example predictions, needed later for significance tests and error analysis.
    with open(os.path.join(out_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r, yp, logit in zip(test, y_pred, pred.predictions):
            f.write(json.dumps({"idx": r["idx"], "label": r["label"], "pred": int(yp),
                                "logits": [float(x) for x in logit]}) + "\n")

    if not args.keep_checkpoint:
        import shutil
        shutil.rmtree(os.path.join(out_dir, "ckpt"), ignore_errors=True)
    print(f"[{run_name}] test macro-F1 = {pred.metrics['test_macro_f1']:.4f}  -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
