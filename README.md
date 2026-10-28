# Stance-aware, LLM-explanation-augmented hate speech detection for replies in conversation

MTech dissertation code. The task is to classify a Reddit **reply** as *hate*, *neutral* or *counter-speech*,
using its **parent comment** as context.

## Phase 1 — baselines (reproducing Yu, Blanco & Hong, NAACL 2022)

| Setting | Model input |
|---|---|
| `target` | reply only |
| `context_target` | `<s> parent </s></s> reply </s>` (sentence pair) |

Dataset: [xinchenyu/counter_context](https://github.com/xinchenyu/counter_context). Gold split: 3,325 / 713 / 713
(train / val / test); there are also 2,094 silver training pairs. `run_baselines.sh` clones it into `data/` automatically.

### Run (Colab / Kaggle GPU)

```bash
pip install -r requirements.txt
bash run_baselines.sh roberta-base             # 2 settings x 5 seeds, prints the mean ± std table
bash run_baselines.sh microsoft/deberta-v3-base   # Phase 2 stronger-encoder baseline
```

To run a single experiment:

```bash
python baseline.py --input context_target --model roberta-base --seed 42
python baseline.py --input context_target --model roberta-base --seed 42 --use_silver
```

Each run writes `outputs/<run>/results.json`, which holds the metrics, per-class report and confusion matrix. It also writes `test_predictions.jsonl`, the per-example predictions used for significance tests and error analysis.
`python aggregate.py` summarises all runs.

### Notes
- Model selection uses gold validation macro-F1 with early stopping (patience 3). Results are reported on gold test.
- Silver data is only ever added to *training*.
- The label ids are mapped as `0=hate, 1=neutral, 2=counter`. This mapping was inferred from the class frequencies and examples, so confirm it against the paper before reporting per-class scores.
- Roughly 10–15 minutes per run on a T4 for roberta-base.
