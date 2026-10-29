"""Phase 3a: ask an instruction-tuned LLM, once, about every gold parent->reply pair.

For each pair it writes one JSON line:
  {"split": "train", "idx": 12, "stance": "support"|"oppose"|"neutral",
   "explanation": "...", "parse_ok": true, "raw": "..."}

The LLM never sees the hate/counter/neutral label and is told not to judge hatefulness, so its
outputs can be used as model input on the test set without leaking labels.

The output file is appended after every batch and existing lines are skipped on restart, so an
interrupted Kaggle session continues where it stopped (copy the earlier file in with --resume_from).

Example (Kaggle, 2 x T4):
  python generate_llm.py --out /kaggle/working/llm/llm_outputs.jsonl
"""

import argparse
import json
import os
import re
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You analyse conversations from Reddit. You answer only with JSON."

PROMPT = """Parent comment:
\"\"\"{context}\"\"\"

Reply to the parent comment:
\"\"\"{target}\"\"\"

Return a JSON object with two keys:
"stance": the reply's stance toward the parent comment, exactly one of "support", "oppose", "neutral".
"explanation": one or two sentences on how the reply relates to the parent comment: what it responds to, who (if anyone) it targets, and any implied meaning or sarcasm.
Do not say whether the reply is hateful or offensive. Return only the JSON object."""

STANCES = ("support", "oppose", "neutral")


def parse(text):
    """Return (stance, explanation, parse_ok). Falls back to a regex when the JSON is malformed."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            stance = str(obj.get("stance", "")).strip().lower()
            expl = " ".join(str(obj.get("explanation", "")).split())
            if stance in STANCES and expl:
                return stance, expl, True
        except json.JSONDecodeError:
            pass
    sm = re.search(r'"?stance"?\s*:\s*"?(support|oppose|neutral)', text, re.I)
    em = re.search(r'"?explanation"?\s*:\s*"(.*?)("\s*[,}]|$)', text, re.S | re.I)
    stance = sm.group(1).lower() if sm else "neutral"
    expl = " ".join((em.group(1) if em else text).split())[:400]
    return stance, expl, False


def load_pairs(data_dir, splits, limit=0):
    pairs = []
    for split in splits:
        with open(os.path.join(data_dir, "gold", f"{split}.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows[:limit] if limit else rows:
            pairs.append({"split": split, "idx": r["idx"],
                          "context": (r.get("context") or "").strip(), "target": (r.get("target") or "").strip()})
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_dir", default="data/counter_context/data")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out", default="llm/llm_outputs.jsonl")
    p.add_argument("--resume_from", nargs="*", default=[], help="earlier output files to continue from")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--max_context_chars", type=int, default=1500)
    p.add_argument("--limit", type=int, default=0, help="only the first N pairs of each split (smoke test)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not os.path.exists(args.out):
        for prev in args.resume_from:
            if os.path.isfile(prev):
                shutil.copyfile(prev, args.out)
                print(f"continuing from {prev}")
                break

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    done.add((o["split"], o["idx"]))

    pairs = load_pairs(args.data_dir, args.splits, args.limit)
    todo = [x for x in pairs if (x["split"], x["idx"]) not in done]
    print(f"{len(pairs)} pairs, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    model.eval()

    def prompt(x):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(context=x["context"][:args.max_context_chars],
                                                          target=x["target"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Similar lengths in a batch waste less padding.
    todo.sort(key=lambda x: len(x["context"]) + len(x["target"]))
    start, n_done, n_bad = time.time(), 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b:b + args.batch_size]
            enc = tok([prompt(x) for x in batch], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for x, text in zip(batch, texts):
                stance, expl, ok = parse(text)
                n_bad += not ok
                f.write(json.dumps({"split": x["split"], "idx": x["idx"], "stance": stance,
                                    "explanation": expl, "parse_ok": ok, "raw": text.strip()},
                                   ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            rate = n_done / (time.time() - start)
            print(f"{n_done}/{len(todo)} done | {rate:.2f} pairs/s | "
                  f"~{(len(todo) - n_done) / rate / 60:.0f} min left | unparsed so far: {n_bad}", flush=True)

    print(f"finished: {n_done} pairs in {(time.time() - start) / 60:.1f} min, {n_bad} needed the fallback parser")


if __name__ == "__main__":
    main()
