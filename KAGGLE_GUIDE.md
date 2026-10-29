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
| 2 | DeBERTa-v3-base, same 10 runs | ✅ code ready | **1–2 h** |
| 3a | LLM explanations (+ stance labels) for every parent→reply pair | ⏳ planned | **2–4 h** (run once, cached) |
| 3b | Classifier with explanations, 2 variants × 5 seeds | ⏳ planned | **1–1.5 h** |
| 4 | Stance multi-task model (λ search + 5 seeds) | ⏳ planned | **1–1.5 h** |
| 5 | CAD + ICHCL generalisation, LLM baselines, error analysis | ⏳ planned | **6–12 h** (spread over 2 weeks) |
| | **Total** | | **≈ 12–22 h**, about 2–3 weeks of quota |

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

## Phase 2: stronger encoder, DeBERTa-v3 (⏱ 1–2 h)

**Notebook:** import `kaggle/phase1_baselines.ipynb` again as a new notebook named `phase2-deberta`. **Attach:** `phase1-baselines`.

1. Do checklist steps 1–5. The "Reusing results from" line must list `phase1-baselines`.
2. In the cell `# Phase 2 (optional)`, remove the `#` in front of `!bash run_baselines.sh microsoft/deberta-v3-base`.
3. Save & Run All.

| Step | Time |
|---|---|
| RoBERTa runs (copied from Phase 1, not retrained) | < 1 min |
| 10 × DeBERTa-v3-base runs | ~1–2 h |

**Check:** DeBERTa parent+reply should beat RoBERTa parent+reply. This becomes the **strong baseline** that Phase 3 must beat.
If you see `nan` loss with DeBERTa, tell me. The fix is to turn off fp16 for that model.

## Phase 3: LLM explanations (⏳ code coming next)

The phase is split into two notebooks, so the expensive LLM step runs **only once**:

- **3a `phase3a-explanations` (⏱ 2–4 h, once):** Qwen2.5-7B-Instruct in 4-bit writes a short explanation of the
  parent→reply relationship for all 4,751 gold pairs. **It also predicts the reply's stance in the same pass**, which saves
  doing a separate run for Phase 4. Results are saved in chunks to `explanations.jsonl`, so if the session stops it continues
  where it left off.
- **3b `phase3b-explained` (⏱ 1–1.5 h):** attach 1, 2 and 3a, then train the classifier with `[parent] [SEP] reply [SEP] explanation`
  (relationship explanation vs reply-only explanation, 5 seeds each).

## Phase 4: stance multi-task (⏳ planned, ⏱ 1–1.5 h)

Attach 1, 2 and 3a. Search λ ∈ {0.3, 0.5, 1.0} on one seed (~30 min), then run the best λ on 5 seeds (~45 min).

## Phase 5: generalisation and analysis (⏳ planned, ⏱ 6–12 h, split over sessions)

| Part | Estimated time |
|---|---|
| CAD: baselines + best model, 3 seeds | 3–5 h |
| ICHCL (Hinglish, MuRIL): baselines + best model | 1–2 h, plus 1–2 h for explanations |
| Zero/few-shot LLM baselines on the test set | 0.5–1 h |
| LoRA fine-tuned 7B LLM (optional) | 2–4 h |
| Significance tests + error analysis | CPU only, minutes |
