# Running every phase on Kaggle

There is one Kaggle notebook per phase. Each notebook attaches the outputs of the earlier ones, so nothing is trained twice.
This guide is updated each time a new phase is committed.

> **Times are estimates** for one Tesla T4. Phase 1 records real training time (`train_min` in `summary.md`);
> use it to rescale the later estimates.

## Weekly GPU budget

Kaggle gives about **30 GPU hours per week**, with at most **12 hours per session**. Planned use:

| Phase | What runs | Status | Estimated GPU time |
|---|---|---|---|
| 1 | RoBERTa: reply only vs parent+reply, 5 seeds each | ✅ done, [results](results/phase1_roberta.md) | **41 min (measured)** |
| 1b | Learning-rate tuning (selected on validation weighted F1) | ✅ done, [results](results/phase1b_tuning_roberta.md): RoBERTa lr 1e-5 | **~70 min (measured)** |
| 2 | DeBERTa-v3-base, same 10 runs | ✅ done, [results](results/phase2_deberta.md): not better than RoBERTa | **~70 min (measured)** |
| 3a | LLM explanations + stance labels for every parent→reply pair | ✅ done, [results](results/phase3a_llm.md): 99.9% parsed | **40 min (measured)** |
| 3–4 | Ablation: base / +LLM / +stance / +LLM+stance × RoBERTa + DeBERTa, 5 seeds | ✅ done, [results](results/phase3_ablation.md): +LLM = +3.1 F1 (RoBERTa) | **~5 h** |
| 4b | LLM-only baselines (zero/few-shot) + significance tests + error analysis | ✅ done, [results](results/phase4b_significance.md) | **~30 min** |
| 5 | CAD + ICHCL generalisation | ⏳ planned | **5–10 h** (spread over 2 weeks) |
| | **Total** | | **≈ 17–27 h**, about 2–3 weeks of quota |

Check how much quota you have left by clicking your profile picture → **Settings → Quotas**.

---

## Checklist for every phase (2 min)

1. **Get the notebook:** download `kaggle/phaseN_*.ipynb` from GitHub. Then on Kaggle, go to **Create → New Notebook → File → Import Notebook**.
2. **Rename it** to `phaseN-...` (for example `phase2-deberta`).
3. **Session options:** set Accelerator to **GPU T4 x2** and switch **Internet on**.
4. **Attach earlier phases:** **+ Add Input → Your Work → Notebooks**, then add every earlier phase notebook.
   Their results are found automatically under `/kaggle/input/*/outputs` and are not retrained.
5. **Quick test (≈ 2 min):** run the cells up to and including "Reusing results from:" with Shift+Enter.
   Check that the GPU shows `Tesla T4` and that the list shows your earlier notebooks.
6. **Start the real run:** **Save Version → Save & Run All (Commit) → Save**. You can then close the browser.
7. **Monitor:** go to Profile → Code → the notebook → the running version → **Logs**.
8. **Results:** once the version shows **Complete**, open the **Output** tab and look at `outputs/summary.md`.

If a session stops (12 h limit or an error), simply run **Save & Run All** again. Only runs without a `results.json` train again.

---

## Phase 1: reproduce the baselines (⏱ 41 min measured: about 2.2 min per reply-only run and 5.2 min per parent+reply run)

**Notebook:** `kaggle/phase1_baselines.ipynb`. **Attach:** nothing.

| Step | Time |
|---|---|
| Checklist steps 1–5 | ~5 min |
| Clone dataset + load model | ~1–2 min |
| 5 × reply-only RoBERTa runs | ~15–30 min |
| 5 × parent+reply RoBERTa runs | ~20–35 min (longer inputs) |
| Summary table | seconds |

**What to check in `summary.md`:**
- `roberta-base_context_target` macro-F1 should be close to the paper's RoBERTa-with-context score (±1–2 points).
- `roberta-base_target` should be **lower**. That shows context helps.
- Compare per-class F1 (`f1_hate`, `f1_neutral`, `f1_counter`) with the paper to confirm the label order.
- Note down `train_min` so the remaining estimates can be rescaled.

➡ Send the `summary.md` table before starting Phase 2.

## Phase 1b: learning-rate tuning (⏱ about 1–1.25 h)

This gives the parent+reply baseline a fair chance to reach the paper's 0.61 weighted F1. Choices are made on the
**validation** set only, and the test set is just reported.

**Notebook:** import `kaggle/phase1_baselines.ipynb` again as a new notebook named `phase1b-tuning`. **Attach:** nothing. **Keep this notebook**; the ablation reuses its runs as the tuned base model.

1. Do checklist steps 1–5.
2. In the cell `# Phase 1: RoBERTa…`, put `#` in front of `!bash run_baselines.sh roberta-base`. The default-setting runs
   aren't needed again; their numbers are saved in `results/phase1_roberta.md`.
3. In the cell `# Phase 1b (optional)`, remove the `# ` in front of the two lines starting with `!test` and `!python tune.py`.
   The `!test` line downloads the dataset.
4. Save & Run All.

| Step | Time (based on Phase 1 timings) |
|---|---|
| Search, reply only: 3 learning rates × 2 seeds × 2.2 min | ~13 min |
| Search, parent+reply: 3 × 2 × 5.2 min | ~31 min |
| Final: best learning rate, 3 more seeds for each setting | ~22 min |

**Result:** `outputs/tuning_roberta-base_weighted_f1.md` shows the validation score of each learning rate and the final test
scores. The new runs have their own folders (for example `..._lr3e-05_selweighted_f1_s13`), so the Phase 1 results are kept.

For DeBERTa later, use the same command with `--model microsoft/deberta-v3-base` (about 2–2.5 h).

## Phase 2: stronger encoder, DeBERTa-v3 (⏱ 1–2 h)

**Notebook:** import `kaggle/phase1_baselines.ipynb` again as a new notebook named `phase2-deberta`. **Attach:** nothing.

1. Do checklist steps 1–5. Put `#` in front of `!bash run_baselines.sh roberta-base` so RoBERTa is not
   retrained here (its results live in `phase1b-tuning`).
2. In the cell `# Phase 2 (optional)`, remove the `#` in front of `!bash run_baselines.sh microsoft/deberta-v3-base`.
3. Save & Run All.

| Step | Time |
|---|---|
| RoBERTa runs (copied from Phase 1, not retrained) | < 1 min |
| 10 × DeBERTa-v3-base runs | ~1–2 h |

**Check:** DeBERTa parent+reply should beat RoBERTa parent+reply. This becomes the **strong baseline** that Phase 3 must beat.
If you see `nan` loss with DeBERTa, tell me. The fix is to turn off fp16 for that model.

## Phase 3a: LLM explanations and stance labels (⏱ about 2–4 h, run once)

**Notebook:** `kaggle/phase3a_llm.ipynb`, named `phase3a-llm`. **Attach:** nothing. This notebook uses **both** T4 GPUs.

Qwen2.5-7B-Instruct reads each of the 4,751 gold parent→reply pairs. For each pair it writes the reply's **stance** toward the
parent (support / oppose / neutral) and a 1–2 sentence **explanation** of how the reply relates to the parent.
It never sees the hate/counter/neutral label and is told not to judge hatefulness, so its outputs can be used on the test set.

1. Do checklist steps 1–3.
2. **Smoke test:** run cells 1–4 interactively (~5–10 min, mostly downloading the 15 GB model). In the 3 printed lines, most should
   have `"parse_ok": true` with a sensible stance and explanation.
3. Save & Run All. The log prints `… pairs/s | ~N min left`, which gives the real time after the first batches.
4. **If the session stops:** open the notebook again, use *Add Input → Your Work* to attach its own last version, and Save & Run All.
   It continues from the saved file.
5. Check the last cell. The parse rate should be high (> 90%), and `counter` replies should lean toward `oppose`.

| Step | Time |
|---|---|
| Model download + loading | ~5–10 min |
| 4,751 pairs (greedy decoding, ≤ 120 new tokens) | ~1.5–3.5 h (estimate; the log shows the real speed) |

If it is too slow, add `--model Qwen/Qwen2.5-3B-Instruct` to the generate command (roughly 2× faster, slightly weaker explanations).

## Phase 3–4: ablation, base / +LLM / +stance / +LLM+stance (⏱ about 2 h per RoBERTa, about 4 h per DeBERTa)

**Notebook:** `kaggle/phase3_ablation.ipynb`, named `phase3-ablation`.
**Attach:** `phase3a-llm` (required), plus `phase1b-tuning` and `phase2-deberta` so finished runs are reused.

| Variant | Model input | Training loss |
|---|---|---|
| base | parent + reply | hate/neutral/counter |
| llm | parent + reply + LLM explanation | hate/neutral/counter |
| stance | parent + reply | hate/neutral/counter + 0.5 × stance (extra head) |
| llm_stance | parent + reply + LLM explanation | both |

With stance weight 0 the stance model is exactly the base model, so any difference comes from the stance signal.

1. Do checklist steps 1–5. The cell must print `LLM file: … 4751 lines`.
2. **Smoke test:** run the smoke-test cell (~3–5 min). It trains all four variants on 64 pairs for 1 epoch and prints a small table.
   The numbers mean nothing; it only has to finish without errors.
3. In the last cell, check `MODELS`: RoBERTa uses `roberta-base=1e-5` (chosen in Phase 1b); set DeBERTa's learning rate after Phase 2.
4. Save & Run All. Seeds run in the outer loop, so if the session stops after 12 h every variant still has some seeds,
   and a rerun with this notebook's last version attached finishes the rest.

| Model | Per seed (4 variants) | 5 seeds |
|---|---|---|
| RoBERTa-base | ~25 min (base may be cached from Phase 1b) | ~1.7–2 h |
| DeBERTa-v3-base | ~50 min | ~3.5–4 h |

**Result:** `outputs/ablation_weighted_f1.md` gives weighted/macro F1 (mean ± std over seeds), the change against base, and
per-class F1 for each model × variant.
If DeBERTa shows `nan` loss, tell me; the fix is to turn off fp16 for that model.
To try other stance weights later, add `--stance_weight 0.3` or `1.0`. Each weight gets its own run folders.

## Phase 4b: LLM-only baselines, significance tests, error analysis (⏱ about 30 min)

**Notebook:** `kaggle/phase4b_eval.ipynb`, named `phase4b-eval`. **Attach:** `phase3-ablation` (required) and `phase3a-llm`.
Uses **both** T4 GPUs (the LLM); the significance tests run on the CPU.

1. Do checklist steps 1–5. The cell must print `Ablation results from: [...]` with the phase3-ablation outputs.
2. **Smoke test:** run the smoke-test cell (~5–10 min, mostly the model download). Most of the 5 printed lines should have
   `"parse_ok": true` and a label.
3. Save & Run All.

| Step | Time |
|---|---|
| Model download + loading (twice, once per baseline) | ~10 min |
| Zero-shot on 713 test pairs | ~3–5 min |
| Few-shot (2 examples per class) on 713 test pairs | ~5–10 min |
| Significance tests + error analysis (CPU) | ~1–2 min |

**Result:** `outputs/significance_weighted_f1.md` contains:
- per model × variant: Δ vs base, seeds better, paired t-test p (+ Holm-corrected) and a paired bootstrap 95% CI and p;
- the LLM-only baselines (weighted/macro F1) compared with each encoder's +LLM model;
- error analysis: per gold class, how many replies the explanation fixes or breaks, confusion before/after, and examples.

## Phase 5: generalisation and analysis (⏳ planned, ⏱ 6–12 h, split over sessions)

| Part | Estimated time |
|---|---|
| CAD: baselines + best model, 3 seeds | 3–5 h |
| ICHCL (Hinglish, MuRIL): baselines + best model | 1–2 h, plus 1–2 h for explanations |
| Zero/few-shot LLM baselines on the test set | 0.5–1 h |
| LoRA fine-tuned 7B LLM (optional) | 2–4 h |
| Significance tests + error analysis | CPU only, minutes |
