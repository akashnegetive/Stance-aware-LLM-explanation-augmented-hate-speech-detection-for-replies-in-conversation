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

---

# Phase 5b: class-weighted loss and context-dependent pairs

Kaggle `phase5b-cad`: RoBERTa-base (lr 1e-5) with `--class_weights balanced` (hate 0.70, neutral 0.67, counter 13.2), 3 seeds,
best epoch by validation weighted F1; same CAD pairs and LLM outputs as above. The significance report also compares every
variant with base on CAD's **context-dependent** test pairs (annotators marked that the label needs the previous content).

## Class-weighted ablation (single runs, mean ± std over 3 seeds)

| variant | weighted F1 | Δ vs base | macro F1 | Δ vs base | hate F1 | neutral F1 | counter F1 |
|---|---|---|---|---|---|---|---|
| base | 67.55 ± 0.85 | – | 50.67 ± 1.15 | – | 67.2 | 72.1 | 12.7 |
| llm (explanation) | 69.64 ± 0.24 | +2.09 | 52.30 ± 1.00 | +1.63 | 70.2 | 73.5 | 13.2 |
| llm_label (hybrid) | 70.91 ± 1.18 | +3.36 | 53.05 ± 3.49 | +2.38 | 71.5 | 74.8 | 12.8 |

Significance vs class-weighted base: + explanation +2.09 weighted F1 (bootstrap [+0.40, +3.82], p = 0.020; 3/3 seeds);
hybrid +3.36 (bootstrap [+1.52, +5.12], p = 0.0002; 3/3 seeds); macro F1 +1.63 (p = 0.26) and +2.38 (p = 0.066).

| system (CAD test) | weighted F1 | macro F1 |
|---|---|---|
| class-weighted base, 3-seed ensemble | 67.60 | 50.01 |
| class-weighted + explanation, 3-seed ensemble | 70.67 | 52.25 |
| **class-weighted hybrid, 3-seed ensemble** | **72.64** | **54.62** |
| Qwen2.5-7B 2-shot | 62.20 | 48.13 |

Class-weighted hybrid vs 2-shot prompting: single runs +8.72 weighted / **+4.92 macro F1 (p = 0.0024)**; ensemble +10.45 /
**+6.49 macro (p = 0.0012)**. Without class weights, macro F1 against 2-shot was level (+0.87, p = 0.59).

## Context-dependent vs other test pairs (Δ = variant − base, mean over seeds; paired bootstrap within each subset)

| model | variant | context-dependent (227): base → variant, Δ (p) | other (855): base → variant, Δ (p) |
|---|---|---|---|
| unweighted | llm (explanation) | 54.06 → 58.35, **+4.28 (0.0008)** | 76.86 → 77.75, +0.89 (0.28) |
| unweighted | label | 54.06 → 53.12, −0.94 (0.54) | 76.86 → 78.32, +1.46 (0.079) |
| unweighted | llm_label | 54.06 → 56.36, +2.30 (0.15) | 76.86 → 79.29, +2.43 (0.010) |
| class-weighted | llm (explanation) | 56.15 → 58.29, +2.15 (0.30) | 75.24 → 77.30, +2.06 (0.028) |
| class-weighted | llm_label | 56.15 → 60.08, +3.93 (0.062) | 75.24 → 78.56, +3.32 (0.0016) |

(weighted F1) LLM alone: 2-shot 50.05 on context-dependent pairs vs 71.29 on the rest; zero-shot 40.69 vs 66.25.

## Interpretation
- **The explanation's gain is concentrated on context-dependent replies**: +4.3 weighted F1 (p < 0.001) on the 227 pairs whose
  label needs the parent comment, against +0.9 (not significant) on the rest. The LLM label alone does not help there (−0.9):
  the explanation of the parent→reply relation is what carries the context. With class weighting the explanation's gain on
  this subset is smaller (+2.2, n.s.) and the hybrid's is +3.9 (p = 0.06); the subset is small (227 pairs).
- Context-dependent pairs are much harder for every system (base 54 vs 77; 2-shot LLM 50 vs 71); RoBERTa + explanation reaches 58.
- **Class weighting recovers the counter class** (F1 0 → about 13) at a cost of about 1–2 weighted F1 for base, and makes the
  fine-tuned hybrid significantly better than few-shot prompting on macro F1 as well (+4.9, p = 0.002).
- The best CAD system is the class-weighted hybrid ensemble (72.64 weighted / 54.62 macro F1).
