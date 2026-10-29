# Phase 3a results: LLM stance and explanations

Qwen2.5-7B-Instruct (fp16, greedy decoding, no fine-tuning) on Kaggle 2 × T4. All 4,751 gold parent→reply pairs
(train, val and test) took 36.7 min (2,410 s for the whole notebook). The LLM never sees the gold labels and is told not to judge hatefulness.

- Parsed cleanly as JSON: **99.9%** (7 pairs needed the fallback parser)
- Output: `llm/llm_outputs.jsonl` (one line per pair: split, idx, stance, explanation)

## LLM stance of the reply toward its parent, by gold label

| Gold label | support | oppose | neutral | Total |
|---|---|---|---|---|
| hate | 125 (10%) | 1,093 (84%) | 90 (7%) | 1,308 |
| neutral | 416 (18%) | 1,270 (54%) | 658 (28%) | 2,344 |
| counter | 54 (5%) | 944 (86%) | 101 (9%) | 1,099 |

## Observations
- The labels are plausible: counter-speech replies mostly oppose their parent (86%).
- Hateful replies also mostly oppose their parent (84%), since they typically attack the parent comment or its author.
  Stance therefore separates neutral from hate/counter but carries little signal for hate vs counter, the hardest pair.
- Expectation for the ablation: the stance auxiliary task may help mainly the neutral class; the free-text explanation
  (who is targeted and how) is the more likely source of gains on hate vs counter.
- Possible follow-up if stance alone does not help: a finer relation label (agrees / disagrees with the argument /
  attacks the author / attacks a group / joke or sarcasm) that splits "oppose".
