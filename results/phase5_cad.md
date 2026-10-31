# Phase 5 results: generalisation to CAD (Contextual Abuse Dataset)

CAD (Vidgen et al., NAACL 2021, CC-BY 4.0), converted with `prepare_cad.py`: parent→reply pairs from CAD's own splits;
identity/affiliation/person-directed abuse → hate, CounterSpeech → counter, Neutral/Slur → neutral; all hate and counter
pairs plus an equal number of sampled neutral pairs (`--neutral_ratio 1`, seed 0).

| split | pairs (hate / neutral / counter) | context-dependent |
|---|---|---|
| train | 2,934 (1,393 / 1,467 / 74) | 530 |
| val | 934 (445 / 467 / 22) | 178 |
| test | 1,082 (499 / 541 / 42) | 227 |

Same method as on Yu et al.: Qwen2.5-7B-Instruct (frozen) explanations, stance and 2-shot labels; RoBERTa-base (lr 1e-5,
best epoch by validation weighted F1), 3 seeds; paired bootstrap over the 1,082 test pairs (5,000 samples).

## Ablation against base (RoBERTa, 3 seeds)

| variant | Δ weighted F1 | seeds better | t-test p (Holm) | bootstrap 95% CI, p | Δ macro F1 (bootstrap p) |
|---|---|---|---|---|---|
| llm (explanation) | +1.91 | 3/3 | 0.095 (0.28) | [+0.46, +3.35], 0.011 | +1.37 (0.0064) |
| llm_label (hybrid) | +2.23 | 3/3 | 0.177 (0.35) | [+0.59, +3.81], 0.0072 | +1.56 (0.0072) |
| label (LLM label) | +0.61 | 2/3 | 0.60 (0.60) | [−0.87, +2.09], 0.43 | +0.40 (0.45) |

## LLM used alone vs fine-tuned (ensembles: mean softmax over 3 seeds; LLM: 1 draw)

| system | weighted F1 | macro F1 |
|---|---|---|
| Qwen2.5-7B zero-shot | 57.46 | 43.47 |
| Qwen2.5-7B 2-shot | 62.20 | 48.13 |
| RoBERTa base | 69.58 | 48.20 |
| RoBERTa + LLM label | 68.74 | 47.60 |
| RoBERTa + LLM explanation | 71.29 | 49.41 |
| RoBERTa hybrid (explanation + label) | 70.65 | 48.95 |

Every fine-tuned variant beats 2-shot prompting on weighted F1 by +6.9 to +8.5 points (paired bootstrap p = 0.0002, single
runs averaged over seeds); macro F1 differences against 2-shot are not significant (see the counter-class issue below).

## Error analysis (RoBERTa, 3-seed ensemble, base vs + explanation)

| gold class | pairs | fixed | broken | net |
|---|---|---|---|---|
| hate | 499 | 43 | 16 | +27 |
| neutral | 541 | 25 | 34 | −9 |
| counter | 42 | 0 | 0 | 0 |

Confusion (rows gold; predicted hate / neutral / counter): hate 341/158/0 → 368/131/0; neutral 113/428/0 → 122/419/0;
counter 18/24/0 → 21/21/0.

## Interpretation
- **The explanation gain generalises**: + explanation and the hybrid improve RoBERTa significantly on CAD
  (+1.9 / +2.2 weighted F1, bootstrap p ≤ 0.011, all 3 seeds), smaller than on Yu et al. (+3.1 / +3.8) but the same direction,
  and again mainly by recovering **hate** replies (hate recall 341 → 368 of 499).
- **The LLM label alone does not transfer** (+0.6, not significant), unlike on Yu et al. (+2.8).
- **Fine-tuning matters more on CAD**: fine-tuned models beat 2-shot prompting by about 7–8.5 weighted F1, whereas on Yu et al. the
  gap was about 1–2 points, plausibly because CAD's abuse categories follow dataset-specific guidelines that fine-tuning learns.
- **Limitation, the counter class**: with 74 training examples (2.5%), no fine-tuned model ever predicts counter (F1 = 0), which
  caps macro F1 and explains why macro F1 against the LLM is level. A class-weighted loss is the planned fix.
- With 3 seeds the paired t-test has little power; the bootstrap over 1,082 test pairs is the main test for CAD.
