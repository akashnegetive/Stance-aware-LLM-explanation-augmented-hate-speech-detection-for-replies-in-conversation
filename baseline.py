"""Phase 1 baselines for context-aware hate / counter-speech detection on replies.

Reproduces the two baselines from Yu, Blanco & Hong (NAACL 2022):
  * target         -> classifier sees only the reply ("target")
  * context_target -> classifier sees parent comment ("context") + reply as a sentence pair

Data: https://github.com/xinchenyu/counter_context  (data/gold/{train,val,test}.jsonl)
Each line: {"idx": int, "label": "0"|"1"|"2", "context": str, "target": str}

Example:
  python baseline.py --input context_target --model roberta-base --seed 42

Ablation features (need an LLM file from generate_llm.py):
  --use_explanation   append the LLM's explanation of the parent->reply relation to the input
  --stance_weight W   multi-task training: extra head predicting the reply's stance toward the parent
                      (LLM labels), loss = CE(hate) + W * CE(stance). W = 0 is exactly the base model.

Runs are cached: if <output_dir>/<run>/results.json (or the same file under any --prev_results
directory) exists with the same training settings, the run is skipped instead of retrained.
"""

import argparse
import json
import math
import os
import random
import shutil

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Label ids as stored in the dataset. Names follow the paper's three classes;
# verify the id -> name mapping against the paper before reporting per-class numbers.
LABEL_NAMES = {0: "hate", 1: "neutral", 2: "counter"}
# Stance of the reply toward its parent comment, as produced by generate_llm.py.
STANCE_IDS = {"support": 0, "oppose": 1, "neutral": 2}


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


def attach_llm_outputs(rows, split, llm):
    """Add the LLM's explanation and stance to each row; every gold pair must have one."""
    missing = [r["idx"] for r in rows if (split, r["idx"]) not in llm]
    if missing:
        raise SystemExit(f"LLM file has no output for {len(missing)} {split} pairs (e.g. idx {missing[:5]}); "
                         "finish generate_llm.py first")
    for r in rows:
        out = llm[(split, r["idx"])]
        r["explanation"] = out["explanation"]
        r["stance"] = STANCE_IDS[out["stance"]]


def read_llm_file(path):
    llm = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                llm[(o["split"], o["idx"])] = o
    return llm


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, input_mode, max_len, use_explanation=False, with_stance=False):
        second = [r["target"] for r in rows]
        if use_explanation:
            # The explanation goes after the reply; truncation removes text from the longer segment first.
            second = [f"{t} {tokenizer.sep_token} {r['explanation']}" for t, r in zip(second, rows)]
        if input_mode == "target":
            self.enc = tokenizer(second, truncation=True, max_length=max_len)
        else:
            # Parent comment first, reply second. Long parents are truncated before the reply.
            self.enc = tokenizer(
                [r["context"] for r in rows],
                second,
                truncation="longest_first",
                max_length=max_len,
            )
        self.labels = [r["label"] for r in rows]
        self.stance = [r["stance"] for r in rows] if with_stance else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        if self.stance is not None:
            item["stance_labels"] = self.stance[i]
        return item


class StanceMultiTaskModel(nn.Module):
    """The unchanged sequence classifier plus a stance head on the first-token representation.

    Hate predictions come from the original classification head, so with stance_weight = 0
    this is exactly the base model; the stance head only adds a training signal.
    """

    def __init__(self, base, num_stance, stance_weight):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.stance_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden, num_stance))
        self.stance_weight = stance_weight

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, stance_labels=None):
        extra = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **extra)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(out.logits, labels)
            if stance_labels is not None:
                stance_logits = self.stance_head(out.hidden_states[-1][:, 0])
                loss = loss + self.stance_weight * nn.functional.cross_entropy(stance_logits, stance_labels)
        return SequenceClassifierOutput(loss=loss, logits=out.logits)


# Arguments that only control where files go; they do not change the trained model.
NON_TRAINING_ARGS = {"data_dir", "output_dir", "prev_results", "keep_checkpoint", "force", "llm_file"}


# Settings that get a tag in the run name when changed from their default,
# so tuning and ablation runs never overwrite the default-setting runs.
TUNABLE_ARGS = {"use_explanation": "expl", "stance_weight": "stance",
                "lr": "lr", "batch_size": "bs", "epochs": "ep", "max_len": "len",
                "warmup_ratio": "wu", "weight_decay": "wd", "patience": "pat", "select_metric": "sel",
                "limit": "lim"}


def training_config(args):
    return {k: v for k, v in vars(args).items() if k not in NON_TRAINING_ARGS}


def make_run_name(args, defaults):
    def tag(k, name):
        v = getattr(args, k)
        if isinstance(v, bool):
            return f"_{name}"
        return f"_{name}{v:g}" if isinstance(v, float) else f"_{name}{v}"

    tags = "".join(tag(k, name) for k, name in TUNABLE_ARGS.items() if getattr(args, k) != defaults[k])
    silver = "_silver" if args.use_silver else ""
    return f"{args.model.replace('/', '_')}_{args.input}{silver}{tags}_s{args.seed}"


def find_cached_run(run_name, args, defaults):
    """Return the run directory of a finished run with identical training settings, if any."""
    for base in [args.output_dir] + list(args.prev_results):
        path = os.path.join(base, run_name, "results.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            saved = json.load(f)
        # Options added after a run was saved are treated as having their default value.
        saved_config = {k: v for k, v in {**defaults, **saved["args"]}.items() if k not in NON_TRAINING_ARGS}
        if saved_config == training_config(args):
            return os.path.dirname(path)
        print(f"[{run_name}] found {path} but its settings differ; ignoring it")
    return None


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "accuracy": float((preds == labels).mean()),
    }


def build_parser():
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
    p.add_argument("--select_metric", choices=["macro_f1", "weighted_f1"], default="macro_f1",
                   help="validation metric used to pick the best epoch (the paper reports weighted F1)")
    p.add_argument("--llm_file", default=None, help="explanations + stance labels from generate_llm.py")
    p.add_argument("--use_explanation", action="store_true", help="append the LLM explanation to the input")
    p.add_argument("--stance_weight", type=float, default=0.0, help="weight of the auxiliary stance loss (0 = off)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N pairs of each split (smoke tests)")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--keep_checkpoint", action="store_true")
    p.add_argument("--prev_results", nargs="*", default=[],
                   help="read-only dirs with earlier outputs (e.g. a Kaggle input); finished runs found there are reused")
    p.add_argument("--force", action="store_true", help="retrain even if a cached result exists")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    defaults = vars(p.parse_args([]))
    uses_llm = args.use_explanation or args.stance_weight > 0
    if uses_llm and not args.llm_file:
        p.error("--use_explanation / --stance_weight need --llm_file")
    if uses_llm and args.use_silver:
        p.error("LLM outputs exist only for gold pairs; do not combine them with --use_silver")

    set_seed(args.seed)
    run_name = make_run_name(args, defaults)
    out_dir = os.path.join(args.output_dir, run_name)

    cached = None if args.force else find_cached_run(run_name, args, defaults)
    if cached:
        if os.path.abspath(cached) != os.path.abspath(out_dir):
            shutil.copytree(cached, out_dir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("ckpt"))
        print(f"[{run_name}] cached result found in {cached}; skipping training")
        return
    os.makedirs(out_dir, exist_ok=True)

    train = read_jsonl(os.path.join(args.data_dir, "gold", "train.jsonl"))
    val = read_jsonl(os.path.join(args.data_dir, "gold", "val.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "gold", "test.jsonl"))
    if args.use_silver:
        # Silver data is only ever added to training; model selection and testing stay on gold.
        train += read_jsonl(os.path.join(args.data_dir, "silver", "train.jsonl"))
        train += read_jsonl(os.path.join(args.data_dir, "silver", "val.jsonl"))
    if args.limit:
        train, val, test = train[:args.limit], val[:args.limit], test[:args.limit]
    if uses_llm:
        llm = read_llm_file(args.llm_file)
        for split, rows in (("train", train), ("val", val), ("test", test)):
            attach_llm_outputs(rows, split, llm)
    print(f"[{run_name}] train={len(train)} val={len(val)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_NAMES),
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )
    # transformers v5 loads weights in the dtype they were saved in (fp16 for DeBERTa-v3); mixed-precision
    # training needs fp32 master weights, otherwise it fails with "Attempting to unscale FP16 gradients".
    model = model.float()
    with_stance = args.stance_weight > 0
    if with_stance:
        model = StanceMultiTaskModel(model, len(STANCE_IDS), args.stance_weight)

    def make_ds(rows):
        return PairDataset(rows, tokenizer, args.input, args.max_len, args.use_explanation, with_stance)

    ds_train, ds_val, ds_test = make_ds(train), make_ds(val), make_ds(test)
    truncated = sum(len(ids) >= args.max_len for ids in ds_train.enc["input_ids"])
    print(f"[{run_name}] {truncated} of {len(ds_train)} training inputs reach max_len={args.max_len} (truncated)")

    # transformers v5 removed `warmup_ratio`; an integer step count works in both v4 and v5.
    total_steps = math.ceil(len(ds_train) / args.batch_size) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "ckpt"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=args.select_metric,
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        # Only the hate labels are evaluation targets; stance_labels is an extra training input.
        label_names=["labels"],
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
    train_output = trainer.train()

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
        "train_runtime_sec": train_output.metrics.get("train_runtime"),
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
        shutil.rmtree(os.path.join(out_dir, "ckpt"), ignore_errors=True)
    print(f"[{run_name}] test macro-F1 = {pred.metrics['test_macro_f1']:.4f}  -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
