"""Phase 5: turn CAD (Contextual Abuse Dataset, Vidgen et al., NAACL 2021) into parent -> reply pairs.

Output has the same format as the Yu et al. data, so every script works with
`--data_dir data/cad/data --dataset cad`:
  data/cad/data/gold/{train,val,test}.jsonl  with  idx, label, context, target (+ cad_id, context_dependent, subreddit)

Pairs: every CAD entry whose parent entry is also in CAD (opening posts have no parent and are skipped).
Labels (CAD primary categories; an entry can have several annotations):
  IdentityDirectedAbuse / AffiliationDirectedAbuse / PersonDirectedAbuse -> 0 hate
  CounterSpeech                                                          -> 2 counter
  Neutral, Slur (non-derogatory use)                                     -> 1 neutral
  precedence: abuse > counter > neutral.
`context_dependent` is true when an annotation says the label needs the previous content (annotation_Context).
Splits: CAD's own train/dev/test (dev -> val). Neutral pairs are downsampled per split to at most
--neutral_ratio x (hate + counter) pairs (all hate and counter pairs are kept); 0 keeps everything.

Example:
  python prepare_cad.py --download --cad_dir data/cad/raw --out_dir data/cad/data
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import statistics
import sys
import urllib.request
import zipfile

ABUSE = {"IdentityDirectedAbuse", "AffiliationDirectedAbuse", "PersonDirectedAbuse"}
ZENODO_RECORD = "https://zenodo.org/api/records/4881008"
NAMES = {0: "hate", 1: "neutral", 2: "counter"}

csv.field_size_limit(sys.maxsize)


def find_full_tsv(cad_dir):
    for name in ("cad_v1_1.tsv", "cad_v1.tsv"):
        hits = sorted(glob.glob(os.path.join(cad_dir, "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


def download(cad_dir):
    """Download Zenodo files, smallest first, until the CAD TSV is found (zip files are extracted)."""
    os.makedirs(cad_dir, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD, timeout=60) as r:
        files = json.load(r)["files"]
    for f in sorted(files, key=lambda f: f["size"]):
        if find_full_tsv(cad_dir):
            return
        path = os.path.join(cad_dir, f["key"])
        print(f"downloading {f['key']} ({f['size'] / 1e6:.1f} MB)", flush=True)
        urllib.request.urlretrieve(f["links"]["self"], path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                z.extractall(cad_dir)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def clean(text):
    text = (text or "").replace("[linebreak]", " ").strip().strip('"')
    return " ".join(text.split())


def entry_splits(full_rows, full_path):
    """info_id -> train/val/test, from a split column or from CAD's split files."""
    norm = {"train": "train", "dev": "val", "val": "val", "test": "test"}
    if "split" in full_rows[0]:
        return {r["info_id"]: norm[r["split"]] for r in full_rows if r.get("split") in norm}
    base = full_path[:-len(".tsv")]
    by_row_id = {r["id"]: r["info_id"] for r in full_rows if "id" in r}
    info_ids = {r["info_id"] for r in full_rows}
    out = {}
    for split, mapped in (("train", "train"), ("dev", "val"), ("test", "test")):
        path = f"{base}_{split}.tsv"
        if not os.path.isfile(path):
            sys.exit(f"no 'split' column and no split file {path}")
        for r in read_tsv(path):
            i = r.get("id", "")
            info = i if i in info_ids else by_row_id.get(i)
            if info:
                out[info] = mapped
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cad_dir", default="data/cad/raw", help="where the CAD TSV files are (searched recursively)")
    p.add_argument("--out_dir", default="data/cad/data")
    p.add_argument("--download", action="store_true", help="download CAD from Zenodo into --cad_dir first")
    p.add_argument("--neutral_ratio", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.download and not find_full_tsv(args.cad_dir):
        download(args.cad_dir)
    full_path = find_full_tsv(args.cad_dir)
    if not full_path:
        sys.exit(f"no cad_v1_1.tsv / cad_v1.tsv under {args.cad_dir}; files there: {os.listdir(args.cad_dir)}")
    rows = read_tsv(full_path)
    print(f"read {len(rows)} annotation rows from {full_path}")
    needed = {"info_id", "info_id.parent", "meta_text", "annotation_Primary"}
    if not needed <= set(rows[0]):
        sys.exit(f"missing columns {sorted(needed - set(rows[0]))}; found {list(rows[0])}")

    # One entry per info_id: text, parent, all primary labels, context flags
    entries = {}
    for r in rows:
        e = entries.setdefault(r["info_id"], {"text": "", "parent": "", "labels": set(), "prev_ctx": False,
                                              "subreddit": r.get("info_subreddit", "")})
        if not e["text"]:
            e["text"] = clean(r["meta_text"])
        parent = (r.get("info_id.parent") or "").strip()
        if parent and parent.lower() not in ("nan", "na", '"na"'):
            e["parent"] = parent
        e["labels"].add(r["annotation_Primary"].strip())
        e["prev_ctx"] |= r.get("annotation_Context", "").strip() == "PreviousContent"
    splits = entry_splits(rows, full_path)
    print(f"{len(entries)} entries; {len(splits)} with a split; primary labels: "
          f"{dict(collections.Counter(l for e in entries.values() for l in e['labels']))}")

    pairs = collections.defaultdict(list)
    skipped = collections.Counter()
    for info_id, e in entries.items():
        if not e["parent"]:
            skipped["no parent (opening post)"] += 1
            continue
        parent = entries.get(e["parent"])
        if parent is None or not parent["text"]:
            skipped["parent not in CAD / empty"] += 1
            continue
        if not e["text"]:
            skipped["empty reply"] += 1
            continue
        if info_id not in splits:
            skipped["no split"] += 1
            continue
        label = 0 if e["labels"] & ABUSE else 2 if "CounterSpeech" in e["labels"] else 1
        pairs[splits[info_id]].append({"cad_id": info_id, "label": label, "context": parent["text"],
                                       "target": e["text"], "context_dependent": e["prev_ctx"],
                                       "subreddit": e["subreddit"]})
    print("skipped entries:", dict(skipped))

    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out_dir, "gold"), exist_ok=True)
    print("\n| split | pairs before sampling (hate / neutral / counter) | kept (hate / neutral / counter) | "
          "context-dependent | reply words (median) | parent words (median) |")
    print("|---|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        ps = sorted(pairs[split], key=lambda x: x["cad_id"])
        before = collections.Counter(x["label"] for x in ps)
        if args.neutral_ratio > 0:
            keep_n = int(args.neutral_ratio * (before[0] + before[2]))
            neutral = [x for x in ps if x["label"] == 1]
            chosen = {x["cad_id"] for x in rng.sample(neutral, min(keep_n, len(neutral)))}
            ps = [x for x in ps if x["label"] != 1 or x["cad_id"] in chosen]
        after = collections.Counter(x["label"] for x in ps)
        with open(os.path.join(args.out_dir, "gold", f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for i, x in enumerate(ps):
                f.write(json.dumps({"idx": i, "label": str(x["label"]), **{k: v for k, v in x.items() if k != "label"}},
                                   ensure_ascii=False) + "\n")
        med = lambda k: statistics.median(len(x[k].split()) for x in ps) if ps else 0
        fmt = lambda c: " / ".join(str(c[k]) for k in (0, 1, 2))
        print(f"| {split} | {sum(before.values())} ({fmt(before)}) | {len(ps)} ({fmt(after)}) | "
              f"{sum(x['context_dependent'] for x in ps)} | {med('target'):.0f} | {med('context'):.0f} |")
    print(f"\nwritten to {args.out_dir}/gold/{{train,val,test}}.jsonl (labels: 0 hate, 1 neutral, 2 counter)")


if __name__ == "__main__":
    main()
