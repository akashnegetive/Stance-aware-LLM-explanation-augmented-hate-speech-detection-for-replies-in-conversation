# Phase 2 results: DeBERTa-v3-base baselines

Run on Kaggle (1 × Tesla T4, transformers 5.16.1) with the same protocol as Phase 1: lr 2e-5, best epoch by validation
macro F1, 5 seeds (13, 42, 87, 123, 2024), gold test set. Training took about 62 min (8.6 min per parent+reply run, 3.9 min per reply-only run).

| setting | n | macro_f1 | weighted_f1 | accuracy | train_min | f1_hate | f1_neutral | f1_counter |
|---|---|---|---|---|---|---|---|---|
| microsoft_deberta-v3-base_context_target | 5 | 54.60 ± 1.27 | 58.20 ± 1.75 | 58.20 ± 2.53 | 8.60 ± 1.74 | 52.63 ± 2.09 | 68.04 ± 3.23 | 43.13 ± 0.89 |
| microsoft_deberta-v3-base_target | 5 | 53.72 ± 0.92 | 57.55 ± 0.73 | 57.70 ± 0.83 | 3.86 ± 0.63 | 52.58 ± 1.89 | 67.94 ± 1.13 | 40.65 ± 1.42 |

## Comparison with RoBERTa-base (Phase 1, same protocol)

| Model | Reply only (weighted / macro F1) | Parent + reply (weighted / macro F1) |
|---|---|---|
| RoBERTa-base | 59.01 / 55.79 | 58.85 / 55.71 |
| DeBERTa-v3-base | 57.55 / 53.72 | 58.20 / 54.60 |

## Observations
- DeBERTa-v3-base does not outperform RoBERTa-base on this dataset (about 1 point lower, within seed variation).
- The parent comment helps DeBERTa slightly (+0.65 weighted F1, +0.9 macro F1, +2.5 counter F1), within seed variation.
- For both encoders, hate (about 52) and counter (40–47) remain the weakest classes.
- The ablation uses DeBERTa at lr 2e-5 (not tuned; RoBERTa proved insensitive to learning rate). Its base variant is
  retrained there with validation weighted-F1 selection, the same protocol as RoBERTa's base.
