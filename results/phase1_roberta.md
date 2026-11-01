# Phase 1 results: RoBERTa-base baselines

Run on Kaggle (1 × Tesla T4, transformers 5.16.1). There were 5 seeds per setting (13, 42, 87, 123, 2024), and all scores are on the gold test set (713 pairs).
Model selection used validation macro-F1 with early stopping (patience 3). Total runtime was about 41 min.

| setting | n | macro_f1 | weighted_f1 | accuracy | train_min | f1_hate | f1_neutral | f1_counter |
|---|---|---|---|---|---|---|---|---|
| roberta-base_context_target | 5 | 55.71 ± 1.44 | 58.85 ± 0.94 | 58.60 ± 0.93 | 5.20 ± 0.65 | 52.64 ± 2.70 | 67.57 ± 1.62 | 46.92 ± 2.93 |
| roberta-base_target | 5 | 55.79 ± 2.01 | 59.01 ± 1.83 | 58.88 ± 1.82 | 2.21 ± 0.30 | 53.76 ± 2.09 | 67.85 ± 1.58 | 45.77 ± 3.65 |

## Comparison with Yu, Blanco & Hong (NAACL 2022)

Paper values are from Yu, Blanco & Hong (NAACL 2022), Table 6 (test split, RoBERTa-base, weighted F1), checked against the paper.

| Paper setting (Table 6) | hate F1 | counter F1 | neutral F1 | weighted F1 |
|---|---|---|---|---|
| Trained with Target | 0.56 | 0.38 | 0.69 | 0.58 |
| Trained with Parent_Target | 0.59 | 0.44 | 0.70 | 0.61 |
| Parent_Target + Silver + stance pretraining (paper's best) | 0.60 | 0.53 | 0.71 | 0.64 (0.63 ± 0.01 over seeds, Table 10) |

Notes from the paper: the plain Parent_Target gain (0.61 vs 0.58) carries **no significance mark** in Table 6 (McNemar's test marks only the Silver / pretraining rows); Table 6 rows appear to be single runs (multiple seeds are reported only in Table 10); training used a fixed number of epochs (5 for Target), batch 16, lr 1e-5, dropout 0.5 and an extra 768-unit tanh layer (Table 11, jiant implementation).

| Setting | Paper | Ours |
|---|---|---|
| Reply only (weighted F1) | 0.58 | 59.01 ± 1.83 |
| Parent + reply (weighted F1) | 0.61 | 58.85 ± 0.94 |
| Parent + reply, per class (hate / neutral / counter) | 0.59 / 0.70 / 0.44 | 52.6 / 67.6 / 46.9 |
| Reply only, per class (hate / neutral / counter) | 0.56 / 0.69 / 0.38 | 53.8 / 67.9 / 45.8 |

## Observations
- The reply-only result reproduces the paper, so the data, training and evaluation pipeline are correct.
- The label mapping `0 = hate, 1 = neutral, 2 = counter` is consistent with the paper's per-class pattern.
- Simply concatenating the parent comment did **not** improve over reply-only here: the difference is within seed variance.
  Context helped *counter* (+1.2 F1) and hurt *hate* (−1.1 F1).
- Truncation is not the cause: only 1 of 4,751 parent+reply pairs exceeds 256 tokens.
- Our parent+reply score is about 2 weighted-F1 points below the paper's single-run 0.61. The paper's own parent+reply gain was not significant, and its setup differs (fixed epochs, dropout 0.5, extra layer); ours is the mean of 5 seeds.
