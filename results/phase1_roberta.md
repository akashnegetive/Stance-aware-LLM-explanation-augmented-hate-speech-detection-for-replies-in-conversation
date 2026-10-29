# Phase 1 results: RoBERTa-base baselines

Run on Kaggle (1 × Tesla T4, transformers 5.16.1). There were 5 seeds per setting (13, 42, 87, 123, 2024), and all scores are on the gold test set (713 pairs).
Model selection used validation macro-F1 with early stopping (patience 3). Total runtime was about 41 min.

| setting | n | macro_f1 | weighted_f1 | accuracy | train_min | f1_hate | f1_neutral | f1_counter |
|---|---|---|---|---|---|---|---|---|
| roberta-base_context_target | 5 | 55.71 ± 1.44 | 58.85 ± 0.94 | 58.60 ± 0.93 | 5.20 ± 0.65 | 52.64 ± 2.70 | 67.57 ± 1.62 | 46.92 ± 2.93 |
| roberta-base_target | 5 | 55.79 ± 2.01 | 59.01 ± 1.83 | 58.88 ± 1.82 | 2.21 ± 0.30 | 53.76 ± 2.09 | 67.85 ± 1.58 | 45.77 ± 3.65 |

## Comparison with Yu, Blanco & Hong (NAACL 2022)

The paper reports weighted F1. Its values below are taken from a secondary summary; verify them against the paper's results table before citing.

| Setting | Paper | Ours |
|---|---|---|
| Reply only (weighted F1) | 0.58 | 59.01 ± 1.83 |
| Parent + reply (weighted F1) | 0.61 | 58.85 ± 0.94 |
| Parent + reply, per class (hate / neutral / counter) | 0.59 / 0.70 / 0.44 | 52.6 / 67.6 / 46.9 |

## Observations
- The reply-only result reproduces the paper, so the data, training and evaluation pipeline are correct.
- The label mapping `0 = hate, 1 = neutral, 2 = counter` is consistent with the paper's per-class pattern.
- Simply concatenating the parent comment did **not** improve over reply-only here: the difference is within seed variance.
  Context helped *counter* (+1.2 F1) and hurt *hate* (−1.1 F1).
- Truncation is not the cause: only 1 of 4,751 parent+reply pairs exceeds 256 tokens.
- Our parent+reply score is about 2 weighted-F1 points below the paper; hyperparameter tuning is still to be done.
