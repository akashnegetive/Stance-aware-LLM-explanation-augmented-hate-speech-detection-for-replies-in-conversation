# Phase 3–4 results: ablation (base / +LLM explanation / +stance / +LLM + stance)

Run on Kaggle (1 × Tesla T4, transformers 5.16.1). All variants use the parent + reply input, best epoch by validation
weighted F1, 5 seeds (13, 42, 87, 123, 2024), gold test set (713 pairs). Within each encoder, every variant uses the
same learning rate: RoBERTa 1e-5 (tuned in Phase 1b) and DeBERTa 2e-5 (Phase 2 default, not tuned). The stance weight is 0.5.
LLM outputs come from Phase 3a: Qwen2.5-7B-Instruct, frozen, never shown the labels.

| model | variant | weighted F1 | Δ vs base | macro F1 | Δ vs base | hate F1 | neutral F1 | counter F1 |
|---|---|---|---|---|---|---|---|---|
| roberta-base (lr 1e-05) | base | 58.86 ± 1.06 | – | 55.14 ± 0.96 | – | 52.0 | 69.1 | 44.3 |
| roberta-base (lr 1e-05) | llm | 61.97 ± 1.01 | +3.12 | 58.45 ± 1.22 | +3.31 | 57.6 | 71.5 | 46.2 |
| roberta-base (lr 1e-05) | stance | 58.88 ± 0.66 | +0.02 | 55.30 ± 0.83 | +0.16 | 55.2 | 68.5 | 42.2 |
| roberta-base (lr 1e-05) | llm_stance | 61.81 ± 0.19 | +2.96 | 58.06 ± 0.20 | +2.92 | 57.9 | 71.9 | 44.3 |
| microsoft/deberta-v3-base (lr 2e-05) | base | 57.88 ± 0.83 | – | 53.83 ± 1.50 | – | 53.8 | 68.8 | 38.8 |
| microsoft/deberta-v3-base (lr 2e-05) | llm | 59.94 ± 2.19 | +2.06 | 56.07 ± 2.83 | +2.25 | 54.6 | 70.5 | 43.1 |
| microsoft/deberta-v3-base (lr 2e-05) | stance | 59.19 ± 0.68 | +1.31 | 55.29 ± 0.60 | +1.46 | 53.0 | 69.9 | 43.0 |
| microsoft/deberta-v3-base (lr 2e-05) | llm_stance | 59.18 ± 1.15 | +1.30 | 54.90 ± 1.73 | +1.07 | 52.6 | 70.9 | 41.2 |

(± is the standard deviation over seeds; per-class F1 is the mean over seeds.)

## Observations
- **LLM explanations help both encoders**: +3.12 weighted / +3.31 macro F1 for RoBERTa and +2.06 / +2.25 for DeBERTa.
  The largest gain is on the hate class (RoBERTa 52.0 → 57.6).
- RoBERTa + LLM (61.97 weighted F1) exceeds the paper's reported parent + reply result (0.61, secondary source; verify)
  and our own parent + reply baseline (58.86).
- **Coarse stance gives little or nothing**: +0.02 for RoBERTa, +1.31 for DeBERTa. This matches Phase 3a, where hate and
  counter replies both mostly "oppose" their parent (84% and 86%), so stance cannot separate the hardest pair.
- Combining stance with the explanation adds nothing over the explanation alone, but RoBERTa's seed variance drops
  sharply (±0.19 vs ±1.01).

## Significance (rough, to be replaced by paired tests)
A Welch t-test from the summary statistics above (n = 5 seeds per variant, unpaired) gives approximately:
RoBERTa +LLM p ≈ 0.003; DeBERTa +LLM p ≈ 0.13; DeBERTa +stance p ≈ 0.04. A proper paired test over seeds, plus
a bootstrap over test examples using `test_predictions.jsonl`, is still needed before claiming significance.

## Next
- Paired significance tests and per-example error analysis (which hate/counter cases the explanation fixes).
- Finer relation labels (agrees / disagrees with the argument / attacks the author / attacks a group / joke or
  sarcasm) in place of the 3-way stance.
- Baselines a reviewer will ask for: the LLM classifying directly (zero/few-shot), and reply + explanation without the parent.
