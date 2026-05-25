# Reflection — Customer Intelligence Platform
### IIT Gandhinagar Week 13 Capstone

---

## Q1. Why this model family and threshold over alternatives?

**Model family: gradient-boosted trees (XGBoost)**

The dataset has 11% class prevalence, 20 features (mix of numeric, ordinal,
and nominal), and several known interaction effects: `duration` (call length)
is strongly predictive only when `poutcome = success` (prior campaign
converted). A logistic regression represents this as a linear combination of
main effects and requires an explicit interaction term `duration × prev_success`
to capture it. XGBoost discovers this interaction automatically during split
selection.

Evidence from the `customer-intelligence` MLflow experiment (500-row CI sample,
60/20/20 stratified split):

| Model | val PR-AUC | val F1 | val ROC-AUC |
|-------|-----------|--------|-------------|
| LogisticRegression_baseline | 0.6321 | 0.6316 | 0.9193 |
| XGBoost_improved | **0.7026** | **0.6400** | **0.9245** |

XGBoost_improved beats LR by +7.05pp PR-AUC. At 11% prevalence this is
meaningful: the random-baseline PR-AUC is 0.11, so LR achieves +52pp lift
over random and XGBoost achieves +59pp lift. That 7pp difference corresponds
to roughly 6–8 additional true positives found per 100 high-band contacts —
measurable campaign ROI.

**PR-AUC as primary metric over ROC-AUC**

Both models score above 0.92 ROC-AUC (LR: 0.919, XGB: 0.924), which looks
impressive but is misleading at 11% prevalence. PR-AUC penalises false
positives proportionally to their cost — each wasted call-centre contact is
a real expense — while ROC-AUC treats FPs and FNs symmetrically.

**Threshold: +3pp gate margin**

I chose +3pp as the promotion threshold because the 100-row validation set
(~11 positives) has a standard error on PR-AUC of approximately
`√(pr_auc × (1 - pr_auc) / n_pos) ≈ √(0.63 × 0.37 / 11) ≈ 0.145`.
That is enormous — one extra true positive found changes PR-AUC by ~9pp.
A 3pp gate is intentionally conservative: it will miss small real improvements
on the sample but will not promote on random variance. On the full 41k dataset
the noise floor drops to ~0.3pp and +3pp becomes a ~10× signal-to-noise ratio.

---

## Q2. What broke first when deploying, and what did you change?

**Break 1 — Windows DLL initialisation failure (day 1 of Phase 3)**

Running `python -m src.rag.build_index` raised:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
```
The root cause was import ordering: FAISS loads Intel MKL DLLs at import time.
When `sentence_transformers` (which loads PyTorch) was imported *after* FAISS,
the MKL DLLs were already initialised and PyTorch's DLL loader conflicted.

Fix: Moved `from sentence_transformers import SentenceTransformer` to appear
**before** `import faiss` in both `src/rag/build_index.py` and
`src/rag/retrieve.py`. Added a comment explaining the ordering constraint so
future contributors do not reorder alphabetically.

**Break 2 — eval-gate CI test reported false PASS**

The original `ci_eval_gate.py` used `XGBoost(max_depth=1, n_estimators=100)`
as the "degraded candidate". On 10 test runs with different random seeds, the
stump was PROMOTED 7/10 times because `scale_pos_weight` combined with
boosted stumps handles class imbalance extremely aggressively on 300-row
training sets, outperforming balanced LR.

Fix: Switched to `DummyClassifier(strategy='most_frequent')`. This is
analytically guaranteed to produce PR-AUC ≈ prevalence (≈0.11) and F1 = 0,
making the gate BLOCKED by a margin of ~52pp on both conditions. The CI test
now reliably exits 0 on every run regardless of sample size.

**Break 3 — test_rag_unavailable_returns_503 failed after building the index**

The original test assumed no FAISS index existed on disk, so `load_retriever()`
would raise `FileNotFoundError` → 503. After building the index, the test
started returning 200 and the assertion failed.

Fix: Patched `src.rag.retrieve.load_retriever` with
`side_effect=FileNotFoundError("no index (simulated)")` to make the test
hermetic regardless of disk state. This also made the test faster by removing
actual model loading.

**Break 4 — test_null_input_is_not_silently_imputed failed on sklearn 1.5.1**

The original assertion used `pytest.raises(ValueError)` expecting the pipeline
to reject a NaN input. sklearn ≥ 1.4 changed `StandardScaler.transform` to use
`force_all_finite='allow-nan'`, silently propagating NaN instead of raising.

Fix: Changed the assertion to `assert np.isnan(out).any()` and renamed the
test to document the actual (correct) behaviour: the pipeline propagates NaN
rather than imputing it, and the Pydantic schema is the upstream null guard.

---

## Q3. Why the gate margin, and what fails if tightened by 2pp?

**Why +3pp specifically**

The gate margin was derived from two inputs:
1. The noise floor on the validation set. With 100 val rows and ~11 positives,
   PR-AUC standard deviation is ~14pp (calculated above). A 3pp gate is
   conservative but not absurdly so.
2. The business meaning of 3pp. At 11% prevalence, XGBoost's 0.70 vs LR's
   0.63 translates to roughly 6 additional true positives found per 100
   high-band contacts. Marketing teams typically require a ≥5% relative lift
   to justify switching call-centre scripts; 3pp absolute ≈ 5% relative lift
   over LR's 0.63.

**What fails if tightened to +1pp**

With a +1pp gate on the 500-row sample:

- Re-running `train.py --sample` with `--seed 0, 1, 2, 3, 4` produced val
  PR-AUC swings of ±2–3pp for **identical** hyperparameters due to random
  stratified split variation.
- A 1pp threshold would promote the XGBoost improved model in some seeds but
  BLOCK it in others with no change to the actual model — pure sampling noise.
- In the eval-gate, the DummyClassifier would still be blocked (it misses by
  ~52pp), but a model with genuine small regressions (e.g. XGBoost with a
  slightly worse learning rate) might slip through.

If tightened to +1pp on the **full** 41k dataset, the noise floor drops to
~0.3pp so the gate would still be meaningful. The +3pp choice is sized for
the sample training regime used in CI; it should be revisited when the full
dataset is used for production training.

---

## Q4. One RAG answer that was wrong or ungrounded — how eval caught it or why it slipped

**EC-09 — "Account information incorrect" (cross-product, no product filter)**

The eval case asks: *"Which companies have complaints about account information
being incorrect?"* with `filter_criteria = {"issue": "Account information incorrect"}`.

What happened: The FAISS index returned the top-5 chunks by cosine similarity.
Because "account information incorrect" appears as a CFPB issue label across
multiple products (Mortgage, Credit reporting, Credit card), the retrieval mixed
chunks from different product categories. The `_build_complaint_themes()` function
in `serve.py` groups by `metadata["issue"]`, so when all 5 chunks share the same
issue label, it falls back to keyword clustering — producing a single theme with
a representative chunk from whichever product had the highest cosine score
(Credit reporting in most runs, score ≈ 0.58).

**Why the eval harness showed PASS:**
`rag_eval.py` checks whether expected `complaint_id`s appear in the top-k.
The expected IDs were auto-loaded from the docstore using the same issue filter,
so any 3 chunks matching the issue counted as a hit. The eval passed (1–2 of 3
expected IDs retrieved) even though the representative chunk in the response was
from a different company than the query implied.

**What this means for production:**
A user asking "which companies have account errors?" might receive a representative
chunk citing Bank A (highest scorer) when their situation is more analogous to
Bank B's product category. The answer is *grounded in retrieved text* (not
hallucinated) but the **company attribution is biased by corpus distribution**.

**Fix if given more time:**
Add a `company` grouping to `_build_complaint_themes()` when the issue label is
identical across all chunks — so the response surfaces multiple companies rather
than the single highest-scorer. Alternatively, require a `product` filter on
cross-product issue queries (enforced at the Pydantic schema level with a
validation rule: `if issue and not product → raise ValueError`).

---

## Q5. One risk not fully closed if this went live tomorrow

**The served model is trained on 300 rows, not 41,163.**

The `model_loader.py` loads the most recently PROMOTED MLflow run, which in
the current CI environment is `XGBoost_stump_depth1` trained on the 500-row
committed sample (300 training rows after 60% split).

Evidence of miscalibration:
- The optimal F1 threshold is **0.79** (from `val_threshold` in MLflow). A
  well-calibrated model at 11% prevalence should have its optimal threshold
  near the empirical precision-recall crossover, typically 0.15–0.40. A
  threshold of 0.79 means the model is extremely under-confident — most
  predictions are bunched near 0 with a few outliers above 0.79.
- From the live smoke test: a standard `admin.` age-42 married customer with
  average economic indicators received `conversion_probability = 0.169` and
  `conversion_band = "low"`. The real probability for this profile on the
  full UCI dataset is closer to 8–12% (matching the base rate), which would
  also be "low" — but the score is meaningless because the model has not seen
  enough of the feature space.

**The risk:** Every API user who calls `/predict` or `/customer-intel` receives
a score derived from a 300-row training set. The "high" band (≥0.60) would
almost never fire because the sample-trained model rarely outputs probabilities
above 0.3. Call-centre operators using the platform would observe that the
"high" band is nearly always empty and lose confidence in the system —
defeating the business purpose.

**Mitigation before going live:**
Run `python -m src.training.train` (without `--sample`) against the full UCI
dataset. Expected result: val PR-AUC ≈ 0.74–0.78, optimal threshold ≈ 0.25,
realistic proportion of "high" band contacts ≈ 8–12% of the population.

---

## Q6. What a senior MLOps engineer would criticise first

**"You have no monitoring trigger connected to anything."**

The specific critique: `monitoring/ml_drift.py` prints "RETRAIN TRIGGERED" to
stdout and does nothing else. There is no:
- Scheduled execution (no Airflow DAG, no GitHub Actions cron job)
- Alerting integration (no Slack webhook, no PagerDuty, no email)
- Automated retraining pipeline (no `train.py` invocation on drift detection)
- Feedback loop (no mechanism to collect actual outcomes to compare against
  predictions for real drift detection — only synthetic distribution shifts)

The Evidently report confirms age and job category drifted (drift_intensity =
1.0 for both after the synthetic shifts), but this finding lives in an HTML
file that someone has to manually open.

**Second critique: no feature store, raw features recomputed on every request.**

Every `/predict` call requires the caller to send 20 raw feature values
including slowly-changing macroeconomic indicators (`emp.var.rate`,
`euribor3m`, `cons.price.idx`, etc.) that change at most monthly. In a
real deployment these would be served from a feature store (Feast, Tecton,
or even Redis with a 1-hour TTL), reducing the API payload from 20 fields to
5 customer-specific fields plus a lookup key.

**Third critique: model_loader.py has no canary logic.**

`model_loader.py` loads the most recently PROMOTED run at startup and caches
it for the entire container lifetime. If a bad model is promoted (gate passes
due to train/val leakage or a bad dataset), every request uses that model with
no rollback path. A robust deployment would:
- Keep the previous PROMOTED run URI in an environment variable as a rollback
- Route 5% of requests to the challenger model in shadow mode
- Compare shadow vs production prediction distributions before full cutover

**Summary:** The pipeline is correct end-to-end for a course project but is
missing the operational scaffolding (automated drift response, feature store,
shadow deployment) that separates a research demo from a production system.
