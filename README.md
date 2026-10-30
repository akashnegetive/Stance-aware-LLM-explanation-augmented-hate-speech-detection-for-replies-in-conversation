# Stance-aware, LLM-explanation-augmented hate speech detection for replies in conversation

MTech dissertation code. The task is to classify a Reddit **reply** as *hate*, *neutral* or *counter-speech*,
using its **parent comment** as context.

| File | Purpose |
|---|---|
| `baseline.py` | Trains and evaluates one run (any encoder; optional LLM explanation input and stance multi-task head) |
| `run_baselines.sh` | Phase 1/2: reply-only vs parent+reply, 5 seeds |
| `tune.py` | Phase 1b: learning-rate search, selected on validation |
| `generate_llm.py` | Phase 3a: LLM explanations + stance labels for every gold pair (resumable) |
| `ablation.py` | Phase 3–4: base / +LLM / +stance / +LLM+stance for each encoder |
| `llm_classify.py` | Phase 4b: LLM-only baseline, zero-/few-shot classification (resumable) |
| `significance.py` | Phase 4b: paired t-tests (Holm), paired bootstrap, error analysis with examples |
| `aggregate.py` | mean ± std table over all finished runs |
| `kaggle/*.ipynb` | One Kaggle notebook per phase; see [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) |

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

### Running on Kaggle

Step-by-step instructions and time estimates for every phase are in [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md). In short:

1. On Kaggle, create a notebook: *File → Import Notebook* → upload `kaggle/phase1_baselines.ipynb`.
2. Under *Settings*, set **Accelerator: GPU T4 x2** and turn **Internet on**. Turning Internet on needs a phone-verified Kaggle account.
3. Click **Save Version → Save & Run All (Commit)**. This runs in the background, so you can close the browser.
   When it finishes, everything in `/kaggle/working/outputs` is saved as that version's output.
4. **Later phases:** in the new notebook, use *Add Input → Your Work →* this notebook. The baseline results then appear under
   `/kaggle/input/<name>/outputs`. Pass that folder via `--prev_results` (the notebook does it automatically) and finished runs
   are copied over instead of retrained.

### Caching (no retraining)
- A run is skipped if `results.json` for the same run name **and the same training settings** already exists in
  `--output_dir` or any `--prev_results` folder. If you change a setting such as `--lr`, the run trains again.
- A run that was interrupted has no `results.json`, so rerunning the script only trains the missing runs.
- Use `--force` to retrain anyway. For hyperparameter sweeps, use a separate `--output_dir`, because a retrained run
  overwrites the folder of the same name.

### Notes
- Model selection uses gold validation macro-F1 with early stopping (patience 3). Results are reported on gold test.
- Silver data is only ever added to *training*.
- The label ids are mapped as `0=hate, 1=neutral, 2=counter`. This mapping was inferred from the class frequencies and examples, so confirm it against the paper before reporting per-class scores.
- Roughly 10–15 minutes per run on a T4 for roberta-base.
