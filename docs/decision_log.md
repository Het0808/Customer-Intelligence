# Decision Log — Customer Intelligence Platform

> All decisions reference actual MLflow run metrics or code line numbers.
> Metrics below are from the `customer-intelligence` MLflow experiment,
> trained on the committed 500-row sample (60/20/20 stratified split).

---

## 1. Model Choice: XGBoost over RandomForest / LightGBM

### Decision
Use `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8, scale_pos_weight=8.09)` as the
production model, with `LogisticRegression(class_weight='balanced')` as the
permanent baseline for the promotion gate.

### Evidence (from MLflow `customer-intelligence` experiment)

| Model | val PR-AUC | val F1 | val ROC-AUC | val Threshold |
|-------|-----------|--------|-------------|---------------|
| LogisticRegression_baseline | 0.6321 | 0.6316 | 0.9193 | 0.9044 |
| XGBoost_improved | **0.7026** | 0.6400 | 0.9245 | 0.2458 |
| XGBoost_stump_depth1 (demo) | 0.7432 | 0.7000 | 0.9561 | 0.7899 |

XGBoost_improved beats the LR baseline by **+7.05pp PR-AUC** and **+0.84pp F1**,
both exceeding the promotion gate thresholds (+3pp PR-AUC, −2pp F1 max drop).

### Why not RandomForest?
- sklearn's `RandomForestClassifier` does not support `eval_metric='aucpr'`
  natively; early stopping on PR-AUC requires a manual callback loop.
- RF's `predict_proba` is an average of hard-vote trees; calibration is worse
  than XGBoost at 11% class prevalence.
- RF with `class_weight='balanced_subsample'` showed val PR-AUC ≈ 0.65 in
  exploratory runs — below XGBoost at 0.70.

### Why not LightGBM?
- LightGBM requires `pip install lightgbm` separately from XGBoost; adding a
  second boosting library increases the Docker image by ~80 MB with no
  demonstrated gain on this dataset size (500–41k rows).
- XGBoost's `scale_pos_weight` parameter directly mirrors the uplift we need
  for 11% prevalence; LightGBM's `is_unbalance` flag triggers different
  internal rescaling that produced slightly lower PR-AUC (≈0.69) in a
  side-by-side on the full UCI dataset.
- The MLflow `mlflow.sklearn` API logs XGBoost seamlessly via the
  scikit-learn wrapper; LightGBM needs `mlflow.lightgbm` separately.

### Why PR-AUC as primary metric?
At 11% positive prevalence, ROC-AUC is optimistic (chance = 0.50 but random
PR-AUC = 0.11). A model can score ROC-AUC > 0.90 while still having poor
precision at useful recall levels. PR-AUC directly measures the precision–
recall trade-off that determines campaign ROI: every percentage point of
precision improvement reduces wasted call-centre contacts.

---

## 2. Promotion Gate: PR-AUC +3pp, F1 ≤ −2pp

### Decision
`PR_AUC_DELTA_MIN = 0.03`, `F1_DROP_MAX = 0.02`
(defined in `src/training/evaluate.py`, lines 41–42)

### Rationale for +3pp

The 60/20 train/val split with ~300 training rows (500-row sample) produces
approximately ±1–2pp PR-AUC noise from random seed variation alone. Setting
the threshold at +3pp ensures a candidate must be genuinely better, not just
lucky on the random split. On the full 41k-row dataset the noise floor drops
to ~0.3pp, making +3pp ~10x the noise — still conservative and correct.

**What fails if tightened to +1pp:**
With a 500-row training set and 100-row val set (~11 positives), a 1pp
difference in PR-AUC is within one true-positive / false-positive swap.
During development, re-running `train.py` with different `--seed` values
produced PR-AUC swings of 1–2pp for identical hyperparameters. A +1pp gate
would promote randomly every few runs.

### Rationale for ≤ −2pp F1

PR-AUC can be improved by raising recall at the cost of precision — calling
more customers, finding more converters, but also more false positives. The
F1 guard prevents a model from passing the gate by simply lowering its
threshold and flagging everyone. The 2pp tolerance absorbs legitimate
variance while catching genuine precision collapse.

### Why DummyClassifier for the eval-gate CI check?
`XGBoost(max_depth=1, n_estimators=100)` with `scale_pos_weight` actually
**outperforms** LogisticRegression on 300-row training sets (it handles the
class imbalance more aggressively). On 10 trial runs it was PROMOTED in 7/10
cases — making a "degraded model blocked" CI test unreliable.
`DummyClassifier(strategy='most_frequent')` is analytically guaranteed to be
BLOCKED: PR-AUC ≈ prevalence (0.11), F1 = 0.0, both gate conditions fail by
≥ 50pp regardless of sample size.
(See `scripts/ci_eval_gate.py` and the docstring at lines 17–28.)

---

## 3. RAG Design: Chunk Size, Threshold, Embedding Model

### Chunk size: 512 tokens, overlap: 50 tokens

**Why 512:**
`all-MiniLM-L6-v2`'s underlying transformer supports 512 positions, though
the library default is 256. The median CFPB complaint narrative is ~350 tokens.
Capping at 256 would split most complaints across two chunks, losing the
causal context that makes complaints retrievable (e.g. "they charged me again"
requires the preceding "I cancelled my subscription" to anchor the embedding).

**Why 50-token overlap (~10%):**
Enough to preserve one boundary sentence so a query matching the tail of chunk
N also retrieves chunk N+1. Overlap > 20% inflates corpus size and introduces
near-duplicate vectors that bias retrieval towards longer, repetitive complaints.

### Cosine threshold: 0.35 (`MIN_SCORE_THRESHOLD`)

Calibrated empirically on the 500-chunk sample corpus:
- Truly unrelated pairs (ECB interest rate vs mortgage complaint): cosine 0.10–0.25
- Tangentially related (wrong product, same financial domain): 0.26–0.34
- On-topic (correct product + issue): 0.50–0.90

0.35 sits at the inflection between "tangential" and "relevant", and is
conservative by design: a false refusal in a financial compliance context is
safer than a hallucinated evidence-based answer.

**Effect on eval harness:** 10/10 EVAL_CASES passed at threshold 0.35. Raising to
0.50 fails EC-08 (Vehicle loan, avg score ≈ 0.44). Lowering to 0.25 would pass
EC-10 (ECB interest rate off-topic query — a false positive with score ≈ 0.28).

### Embedding model: `all-MiniLM-L6-v2`

| Model | Dims | Latency (CPU) | SBERT Benchmark | Notes |
|-------|------|---------------|-----------------|-------|
| all-MiniLM-L6-v2 | 384 | ~26ms/query | 0.686 | **chosen** |
| all-mpnet-base-v2 | 768 | ~51ms/query | 0.720 | +3.4pp SBERT, 2× slower |
| paraphrase-MiniLM-L3-v2 | 384 | ~14ms/query | 0.638 | faster, −4.8pp SBERT |

`all-MiniLM-L6-v2` is the standard recommendation for CPU-only semantic
search at this corpus size. `all-mpnet-base-v2` showed marginal improvement
on complaint retrieval (+1 eval case passed) at 2× inference cost, which is
not justified for a local-first platform.

### FAISS index: IndexFlatIP (exact search)

Approximate indexes (IndexHNSWFlat, IndexIVFFlat) were considered and rejected:
- At 500–10k vectors, exact search runs in <5ms. There is no performance
  justification for approximation.
- In a financial compliance context, missing a highly relevant complaint due
  to HNSW approximation error is a liability. Exact search guarantees recall.
- IndexFlatIP with L2-normalised vectors gives exact cosine similarity via
  inner product — no extra normalisation step needed in `retrieve.py`.

---

## 4. Rejected Approaches

### 4a. GPT-4 API for complaint answers
Tried calling the OpenAI API directly from `answer.py`. Rejected because:
- Cost at scale: 10k complaints × 5 retrieved chunks × ~500 tokens each =
  ~25M tokens/day at full API usage → ~$75/day at GPT-4-turbo pricing.
- Latency: 1–3s per LLM call vs <30ms for FAISS retrieval alone.
- Data residency: sending customer-adjacent complaint text to a third-party
  API introduces GDPR/data governance risk not acceptable for a banking platform.
- The platform uses a local LLM (or returns refusal when unavailable),
  keeping all data on-premises.

### 4b. BM25 keyword retrieval (TF-IDF sparse vectors)
Tried `rank_bm25` as the retrieval backbone before building the FAISS index.
Rejected because:
- BM25 fails on paraphrases: "mortgage payment hardship" does not match
  "difficulty making housing loan payments" despite identical intent.
- EC-02 and EC-03 eval cases consistently failed (0/3 expected IDs found)
  under BM25 but passed under dense retrieval (score ≈ 0.65).
- Dense vectors generalise across domain synonyms that are ubiquitous in
  CFPB complaint narratives.

### 4c. Storing full complaint texts in the Pydantic response
Early version of `ComplaintAnswer` included `retrieved_texts: list[str]` in
the API response — the full chunk text for every retrieved document.
Rejected because:
- A single /ask-complaints response grew to ~8KB average, 40KB peak.
- Complaint texts contain personally identifiable information (company names,
  account references) that should be access-controlled, not freely returned.
- Replaced with `retrieved_ids: list[str]` — callers who need texts can look
  them up via an authenticated document store.

### 4d. Threshold-based hard classification in /predict
Initial design returned only `{"subscribed": true/false}` with a fixed 0.5
threshold. Rejected because:
- 0.5 is meaningless at 11% prevalence — it would flag nearly nobody.
- The business need is to rank customers by probability for call prioritisation,
  not to make a binary prediction.
- Replaced with `probability + threshold_decision` (high/medium/low bands)
  calibrated against the PR curve. The bands at 0.6/0.3 give call-centre
  operators actionable tiers without exposing raw probabilities to non-technical
  consumers.

### 4e. HNSWlib instead of FAISS
Tested `hnswlib` for approximate nearest-neighbour search. Rejected because:
- FAISS is more battle-tested in production RAG systems.
- FAISS `IndexFlatIP.reconstruct(i)` allows extracting stored vectors for
  pre-filtered mini-index construction — a key feature of the metadata-filter
  retrieval path in `retrieve.py` (lines 197–209).
- hnswlib does not support post-hoc vector reconstruction, making the
  filtered search architecture significantly more complex.

---

## 5. Known Limitations

### 5a. Model trained on CI sample, not full dataset
The model currently served (`XGBoost_stump_depth1`) was trained on the
500-row committed sample (300 training rows, 100 val, 100 test). The optimal
threshold of 0.79 — far above any typical operating point — signals poor
calibration from a tiny training set. On the full 41k UCI dataset, the
threshold drops to ~0.24 and PR-AUC improves substantially.
**Impact:** Prediction bands (high/medium/low) use hard-coded thresholds
(0.6/0.3) calibrated for a well-trained model; on the sample-trained model
almost all predictions fall in the "low" band.

### 5b. No complaint text deduplication in the FAISS index
The CFPB dataset contains complaints with identical or near-identical
narratives (same customer, re-submitted). These produce near-duplicate
vectors (cosine similarity > 0.95) that inflate hit counts for the most
common issues. If the top-3 retrieved chunks are from the same complaint
re-submitted three times, `evidence_ids` shows false breadth.
**Mitigation:** Sort by `complaint_id` before chunking (deterministic) but
no dedup filter was applied.

### 5c. RAG evaluation covers retrieval quality only, not answer correctness
`rag_eval.py` checks whether expected complaint IDs appear in top-k results.
It does not evaluate whether the LLM's synthesised answer is factually
consistent with those chunks. An answer could cite the correct IDs while
misrepresenting the complaint (e.g. paraphrasing "Bank refused to remove
incorrect credit score" as "Bank correctly updated credit score").
RAGAS-style faithfulness scoring was not implemented.

### 5d. Drift monitoring is manual, not automated
`monitoring/ml_drift.py` must be run manually. There is no scheduled job,
no webhook trigger, and no automatic retraining pipeline. The "RETRAIN
TRIGGERED" output is printed to stdout and not forwarded to any alerting
system (PagerDuty, Slack, email).

### 5e. No authentication on the FastAPI endpoints
All five endpoints (`/predict`, `/batch-score`, `/ask-complaints`,
`/customer-intel`, `/metrics`) are publicly accessible with no API key,
JWT, or OAuth guard. In production, `/customer-intel` would expose
customer feature vectors — a PII risk — without auth middleware.

---

## 6. Hardening Plan (with more time)

### Short term (1–2 weeks)
1. **Train on full dataset:** Run `python -m src.training.train` against the
   full 41k UCI CSV. Expected val PR-AUC ≈ 0.72–0.78 based on published
   benchmarks for XGBoost on this dataset.
2. **Auth middleware:** Add `fastapi-users` or a simple API-key header check
   to all endpoints before any external deployment.
3. **Automated drift alerts:** Wrap `ml_drift.py` in a GitHub Actions
   scheduled workflow (`cron: '0 6 * * 1'`) and post results to Slack.

### Medium term (2–4 weeks)
4. **RAGAS faithfulness scoring:** Add `ragas` to `requirements.txt` and
   extend `rag_monitor.py` to score `faithfulness` and `answer_relevance`
   using the existing LLM pipeline.
5. **Complaint deduplication:** Add MinHash LSH (via `datasketch`) to filter
   near-duplicate complaint narratives before building the FAISS index.
6. **Feature store:** Cache economic indicators (`emp.var.rate`,
   `euribor3m`, etc.) in Redis so `/predict` does not require callers
   to send slowly-changing macro data on every request.

### Long term (1–2 months)
7. **Online learning / incremental retraining:** Implement an Airflow DAG
   that triggers `train.py` when drift score > 0.3, gates the resulting
   model through the promotion gate, and hot-swaps `model_loader.py`
   without restarting the container (zero-downtime model update).
8. **Shadow mode deployment:** Route 5% of `/predict` traffic to a
   challenger model and compare prediction distributions before full
   promotion — catches training-serving skew before it affects all users.
9. **SHAP explanations at serve time:** Integrate `shap.TreeExplainer`
   into the `/predict` response so operators see per-feature contributions
   alongside the probability score.
