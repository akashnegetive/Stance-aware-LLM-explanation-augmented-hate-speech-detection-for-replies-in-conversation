# Phase 1b results: RoBERTa-base learning-rate tuning

Run on Kaggle (1 × Tesla T4, transformers 5.16.1). The best epoch was chosen by validation weighted F1.
The search used 2 seeds (13, 42) per learning rate, and the best learning rate per input was then run on 5 seeds (13, 42, 87, 123, 2024).
**Selection used the gold validation set only**; test scores are reported for the selected configuration.
Training time was about 61 min in total (about 70 min with loading and evaluation).

## Selection (mean validation weighted F1 over the 2 search seeds)

| lr | reply only | parent + reply |
|---|---|---|
| 1e-5 | 58.31 | **59.67 (selected)** |
| 2e-5 | 58.30 | 58.85 |
| 3e-5 | **58.86 (selected)** | 59.21 |

## Selected configurations on the test set (5 seeds)

| setting | lr | weighted F1 | macro F1 | hate F1 | neutral F1 | counter F1 |
|---|---|---|---|---|---|---|
| reply only | 3e-5 | 59.11 ± 1.19 | 55.22 ± 1.73 | 52.14 ± 3.17 | 69.82 ± 2.02 | 43.71 ± 3.53 |
| parent + reply | 1e-5 | 58.86 ± 1.06 | 55.14 ± 0.96 | 51.97 ± 0.98 | 69.15 ± 1.60 | 44.29 ± 2.22 |

## All runs (test set; configurations not selected have only the 2 search seeds)

| setting | n | macro_f1 | weighted_f1 | accuracy | train_min | f1_hate | f1_neutral | f1_counter |
|---|---|---|---|---|---|---|---|---|
| roberta-base_context_target_lr1e-05_selweighted_f1 | 5 | 55.14 ± 0.96 | 58.86 ± 1.06 | 58.99 ± 1.04 | 5.10 ± 0.25 | 51.97 ± 0.98 | 69.15 ± 1.60 | 44.29 ± 2.22 |
| roberta-base_context_target_lr3e-05_selweighted_f1 | 2 | 56.87 ± 1.32 | 59.95 ± 1.05 | 59.54 ± 0.77 | 4.68 ± 0.92 | 52.18 ± 1.30 | 68.66 ± 0.32 | 49.76 ± 2.35 |
| roberta-base_context_target_selweighted_f1 | 2 | 55.19 ± 1.14 | 59.62 ± 0.89 | 61.15 ± 0.42 | 4.05 ± 0.30 | 53.61 ± 0.12 | 71.68 ± 0.31 | 40.27 ± 3.23 |
| roberta-base_target_lr1e-05_selweighted_f1 | 2 | 55.53 ± 0.27 | 59.29 ± 0.29 | 59.47 ± 0.56 | 2.20 ± 0.01 | 51.92 ± 0.97 | 69.71 ± 0.29 | 44.97 ± 0.46 |
| roberta-base_target_lr3e-05_selweighted_f1 | 5 | 55.22 ± 1.73 | 59.11 ± 1.19 | 59.94 ± 0.99 | 1.86 ± 0.21 | 52.14 ± 3.17 | 69.82 ± 2.02 | 43.71 ± 3.53 |
| roberta-base_target_selweighted_f1 | 2 | 57.32 ± 1.77 | 60.35 ± 1.86 | 60.24 ± 1.89 | 2.05 ± 0.14 | 56.07 ± 0.25 | 68.60 ± 2.22 | 47.29 ± 2.85 |

(Rows without an `lr` tag used the default lr 2e-5.) Some configurations that were not selected score higher on test,
but choosing them by test score would be test-set selection, so they are not used.

## Observations
- Validation differences between learning rates are below 1 point, smaller than seed variation (about ±1), so RoBERTa
  is not sensitive to learning rate here. The Phase 1 baseline was already close to its best.
- With tuning, parent + reply (58.86) still does not beat reply only (59.11): plain concatenation of the parent
  comment gives no gain. This motivates the stance and LLM-explanation variants.
- The gap to the paper's reported 0.61 weighted F1 for parent + reply remains (about 2 points); the reply-only result matches the paper.
- **The ablation uses `roberta-base` with lr 1e-5** (the configuration selected on validation for parent + reply).
