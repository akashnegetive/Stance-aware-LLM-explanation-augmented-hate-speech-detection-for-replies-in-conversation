# Phase 4c results: few-shot robustness and the hybrid model

Kaggle `phase4c-hybrid`. Hybrid = RoBERTa-base (lr 1e-5) fine-tuned with the 2-shot LLM's predicted label appended to its input
(`parent </s></s> reply </s> [explanation </s>] LLM label: X`). The LLM (Qwen2.5-7B-Instruct, frozen) labelled all
4,751 gold pairs with the same 2-shot prompt; it never sees a pair's gold label, apart from the 6 shot examples themselves.
5 seeds per variant; best epoch by validation weighted F1; gold test set (713 pairs).
The 4-shot runs produced no results in this run (to be checked).

## RoBERTa ablation including the hybrid

| variant | weighted F1 | Δ vs base | macro F1 | Δ vs base | hate F1 | neutral F1 | counter F1 |
|---|---|---|---|---|---|---|---|
| base | 58.86 ± 1.06 | – | 55.14 ± 0.96 | – | 52.0 | 69.1 | 44.3 |
| llm (explanation) | 61.97 ± 1.01 | +3.12 | 58.45 ± 1.22 | +3.31 | 57.6 | 71.5 | 46.2 |
| label (LLM label) | 61.65 ± 1.05 | +2.79 | 57.58 ± 1.15 | +2.44 | 60.4 | 72.4 | 40.0 |
| **llm_label (explanation + label)** | **62.66 ± 1.16** | **+3.81** | **59.18 ± 1.69** | **+4.04** | 59.5 | 72.0 | 46.0 |

## Significance against base (RoBERTa, 5 seeds; Holm across 8 comparisons; bootstrap 5,000 samples)

| variant | Δ weighted F1 | seeds better | t-test p (Holm) | bootstrap 95% CI, p | Δ macro F1 (bootstrap p) |
|---|---|---|---|---|---|
| llm_label | +3.81 | 5/5 | 0.0095 (0.057) | [+1.25, +6.37], 0.0016 | +4.04 (0.0072) |
| llm | +3.12 | 5/5 | 0.0208 (0.104) | [+0.78, +5.37], 0.0132 | +3.31 (0.0156) |
| label | +2.79 | 5/5 | 0.0034 (0.024) | [+0.13, +5.47], 0.0396 | +2.44 (0.1008) |
| llm_stance | +2.96 | 5/5 | 0.0027 (0.022) | [+0.43, +5.47], 0.0248 | +2.92 (0.0436) |
| stance | +0.02 | 2/5 | 0.98 (0.98) | [−1.55, +1.54], 0.95 | +0.16 (0.83) |

DeBERTa rows are unchanged from Phase 4b ([phase4b_significance.md](phase4b_significance.md)). Bootstrap p-values for variants
already in Phase 4b differ slightly from that report because of bootstrap sampling; the conclusions are the same.

## LLM used alone

| LLM baseline | example sets | weighted F1 | macro F1 | per draw (weighted F1) |
|---|---|---|---|---|
| Qwen2.5-7B zero-shot | 1 | 57.23 | 53.90 | 57.2 |
| Qwen2.5-7B 2-shot | 4 | 61.29 ± 0.63 | 56.73 ± 0.78 | 61.3, 60.6, 62.3, 61.0 |

| fine-tuned (RoBERTa) | Δ vs zero-shot: weighted / macro (p) | Δ vs 2-shot (mean of 4 draws): weighted / macro (p) |
|---|---|---|
| +llm_label | +5.43 (0.0012) / +5.27 (0.0036) | +1.37 (0.30) / +2.45 (0.084) |
| +llm | +4.74 (0.0100) / +4.54 (0.0200) | +0.68 (0.70) / +1.72 (0.35) |
| +label | +4.42 (0.0092) / +3.68 (0.0408) | +0.35 (0.73) / +0.85 (0.45) |

## Interpretation
- Few-shot prompting is stable across example sets (61.29 ± 0.63); the Phase 4b draw (61.27) was typical.
- The hybrid (explanation + LLM label) is the best system: +3.81 weighted / +4.04 macro F1 over the fine-tuned baseline, the largest
  and most significant gain (bootstrap p = 0.0016; t-test p = 0.0095, 0.057 after Holm).
- The two LLM inputs are complementary: the LLM label helps hate (60.4) but hurts counter (40.0); the explanation keeps counter (46.0).
- The hybrid significantly beats zero-shot prompting (+5.4) and is numerically above 2-shot prompting (+1.4 weighted, +2.5 macro),
  but that margin is not significant (macro p = 0.084): with 713 test pairs, 1–2 point differences are hard to establish.

## Next
- Ensemble comparison: 5-seed ensemble of the hybrid vs majority vote of the 4 few-shot draws (CPU only, existing predictions).
- Re-run the 4-shot baseline (missing here).
