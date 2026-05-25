# Demo Script — Customer Intelligence Platform
### 6-Minute Screen Recording Guide

> **Setup before recording:**
> 1. `docker compose up -d` — server running in background
> 2. `mlflow ui` — MLflow UI open at `http://localhost:5000`
> 3. Three terminal tabs ready: server logs, curl commands, Python scripts
> 4. Browser tabs: `http://localhost:8000/docs`, MLflow UI, `monitoring/reports/ml_drift_report.html`
> 5. Font size ≥ 18pt in terminals; zoom browser to 110%

---

## Minute 1 — Repo clone + data validation (live terminal)

**Narration:** "I'll start by showing the full automated validation pipeline
running end-to-end on the committed sample."

```bash
# Show the sample exists and is committed
ls data/samples/
# → bank_marketing_sample.csv  cfpb_complaints_sample.csv

# Run schema validation — this is the same step that runs in CI
python -m src.data_pipeline.validate --sample
# Expected: "Validation PASSED: 500 rows, 21 columns, 0 errors"

# Show that a bad record is rejected
python -m src.data_pipeline.validate --sample --inject-bad
# Expected: exit 1 with "Schema validation FAILED: age=150 out of range [17, 98]"

# Show the GitHub Actions badge on GitHub
# (navigate to the repo page — point to the green CI checkmark)
```

**What to show:** Both validation commands. Linger on the failure output —
this proves the validator actively rejects bad data, not just passes everything.

---

## Minute 2 — MLflow showing both models + gate decision

**Narration:** "The promotion gate is the safeguard that prevents a worse model
from replacing the baseline. I'll show you the MLflow comparison table and the
actual gate output."

```bash
# Open MLflow UI (already running)
# Navigate to: http://localhost:5000
# Experiments → customer-intelligence
# Click "Compare" on LogisticRegression_baseline and XGBoost_improved
```

**In MLflow UI, point to:**
- val_pr_auc column: LR = 0.6321, XGBoost = 0.7026 (delta = +7.05pp > +3pp gate)
- val_f1 column: LR = 0.6316, XGBoost = 0.6400 (delta = +0.84pp, within −2pp tolerance)
- Tags on XGBoost run: `gate_decision = PROMOTED`

```bash
# Show the gate running live (re-run in terminal)
python -m src.training.train --sample --include-stump 2>&1 | grep -A 15 "PROMOTION GATE"
```

**Expected terminal output to show:**
```
==============================================================
  PROMOTION GATE: XGBoost_improved vs LogisticRegression
  PR-AUC  |  Baseline: 0.6321  |  Candidate: 0.7026  |  +0.0705
  F1      |  Baseline: 0.6316  |  Candidate: 0.6400  |  +0.0084
  [OK] PROMOTED -- PR-AUC +0.0705 (>= +0.03 threshold)
==============================================================
  PROMOTION GATE: XGBoost_stump_depth1 vs LogisticRegression
  [X] BLOCKED  -- PR-AUC delta ... | F1 regressed ...
```

**What to show:** The PROMOTED/BLOCKED contrast side-by-side.

---

## Minute 3 — /predict: valid input, then invalid input (422)

**Narration:** "The prediction endpoint accepts a validated customer feature
vector and returns a probability score with a business-tier band."

```bash
# Tab 1: Show server is running
docker compose logs ml-service --tail=3

# Tab 2: Valid prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "duration": 300,
    "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp.var.rate": -1.8, "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
  }' | python -m json.tool
```

**Expected output:**
```json
{
  "probability": 0.169017,
  "threshold_decision": "low",
  "model_version": "XGBoost_stump_depth1",
  "run_id": "...",
  "latency_ms": 12.4,
  "request_id": "..."
}
```

```bash
# Explain the bands: ≥0.6 high, 0.3-0.6 medium, <0.3 low

# Invalid input — age out of range
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 150, "job": "admin.", "marital": "single",
       "education": "high.school", "default": "no", "housing": "no",
       "loan": "no", "contact": "cellular", "month": "jun",
       "day_of_week": "tue", "duration": 120, "campaign": 1,
       "pdays": 999, "previous": 0, "poutcome": "nonexistent",
       "emp.var.rate": -1.8, "cons.price.idx": 93.994,
       "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0}' \
  | python -m json.tool
```

**Expected output (HTTP 422):**
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be less than or equal to 98",
      "input": 150
    }
  ]
}
```

**What to show:** Contrast the 200 vs 422. Mention that Pydantic v2 rejects
invalid inputs before they ever reach the model.

---

## Minute 4 — /ask-complaints: evidence IDs, then refusal

**Narration:** "The RAG endpoint retrieves grounded evidence from CFPB
complaints. I'll show a successful evidence-based retrieval, then deliberately
ask an off-topic question to trigger the refusal gate."

```bash
# Evidence-based answer — mortgage payment difficulties
curl -s -X POST http://localhost:8000/ask-complaints \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What problems do customers report when making mortgage payments?",
    "product": "Mortgage"
  }' | python -m json.tool
```

**Expected output (point to these fields):**
```json
{
  "answer": "Customers report several recurring issues during the mortgage payment process...",
  "retrieved_ids": ["1002029", "1003814", "1005183", "1001744", "1004622"],
  "evidence_sufficiency": "sufficient",
  "prompt_version": "v1",
  "refusal": false
}
```

**Explain:** "Notice `retrieved_ids` — these are the actual CFPB complaint IDs
the answer is based on. The LLM cannot make up evidence; it can only synthesise
from these specific documents."

```bash
# Off-topic refusal — ECB interest rate (no complaint corpus evidence)
curl -s -X POST http://localhost:8000/ask-complaints \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the current European Central Bank interest rate policy and how does it affect euro-area inflation?"
  }' | python -m json.tool
```

**Expected output:**
```json
{
  "answer": null,
  "retrieved_ids": [],
  "evidence_sufficiency": "insufficient",
  "prompt_version": "v1",
  "refusal": true
}
```

**Explain:** "The cosine similarity between this query and every complaint chunk
was below 0.35. The system refuses rather than hallucinating — `answer: null`,
`retrieved_ids: []`. This is by design."

---

## Minute 5 — /customer-intel combining both services

**Narration:** "The integration endpoint is the centrepiece — one request gets
both the ML conversion score and the most relevant CFPB complaint themes for
this customer profile, in a single round trip."

```bash
curl -s -X POST http://localhost:8000/customer-intel \
  -H "Content-Type: application/json" \
  -d '{
    "customer_features": {
      "age": 42, "job": "admin.", "marital": "married",
      "education": "university.degree", "default": "no",
      "housing": "yes", "loan": "no", "contact": "cellular",
      "month": "may", "day_of_week": "mon", "duration": 300,
      "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
      "emp.var.rate": -1.8, "cons.price.idx": 93.994,
      "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
    },
    "product": "Mortgage",
    "issue": null,
    "date_from": null
  }' | python -m json.tool
```

**Expected output (annotate each section):**
```json
{
  "conversion_band": "low",           ← ML output
  "conversion_probability": 0.169017, ← ML output
  "model_version": "XGBoost_stump_depth1",
  "complaint_themes": [               ← RAG output: 4 themes
    {
      "theme": "Trouble during payment process",
      "evidence_ids": ["1005183", "1002029"],
      "representative_chunk": "Product: Mortgage | Issue: Trouble during payment process..."
    },
    ...
  ],
  "index_version": "20260525T111506Z", ← FAISS index build timestamp
  "latency_ms": 90.15                  ← end-to-end latency
}
```

```bash
# Show the JSONL audit log updating
tail -1 logs/customer_intel.jsonl | python -m json.tool
```

**What to emphasise:** "The themes come entirely from CFPB corpus metadata —
no LLM invented these categories. The `index_version` timestamp means we
always know which version of the complaint corpus backed this response."

---

## Minute 6 — Drift report (HTML) + RAG monitoring JSON

**Narration:** "Finally, I'll show the monitoring layer — both the ML feature
drift report and the RAG retrieval metrics."

### ML Drift (browser)

```bash
# Open the HTML report
start monitoring/reports/ml_drift_report.html   # Windows
# or: open monitoring/reports/ml_drift_report.html  (macOS)
```

**In the browser, navigate to:**
1. The Data Drift summary table — show `age` column flagged with drift detected
2. The distribution comparison chart for `age` — reference (peak at 35–45)
   vs current (shifted right by 10, peak at 45–55)
3. The `job` category drift — "blue-collar" bar present in reference,
   absent in current (relabelled to "services")

```bash
# Show the retrain trigger output
python monitoring/ml_drift.py 2>&1
# Expected: "*** RETRAIN TRIGGERED *** · Features: age, job"
```

### RAG Metrics (terminal)

```bash
# Run the RAG monitor live
python monitoring/rag_monitor.py 2>&1
```

**Expected output:**
```
Running 10 eval queries against faiss_index ...

  EC-01 [PASS]  Debt collection communication tactics  (50 ms)
  EC-02 [PASS]  Mortgage payment trouble  (25 ms)
  ...
  EC-10 [PASS]  Off-topic query -- should be refused  (27 ms)

====================================================
  RAG Monitor -- Customer Intelligence Platform
====================================================
  Queries run                   10
  Retrieval hit rate            70.0%
  Refusal rate                  10.0%
  Avg top-1 cosine score        0.6154
  Avg latency (ms)              26.7
====================================================
```

```bash
# Show the saved JSON
python -m json.tool monitoring/reports/rag_metrics.json
```

**Closing line:** "10/10 eval cases pass, average retrieval latency 27ms,
and the drift monitor would trigger a retrain if deployed on a live feature
stream with demographic shift. That's the full system — ingestion, training
with a gated promotion, dual-pipeline serving, CI, and monitoring."

---

## Technical tips for recording

- Use **OBS Studio** (free) set to 1920×1080, 30fps
- Record audio — narration matters for grading technical depth
- Keep the cursor near the output being described
- Don't rush the JSON outputs — let the grader read them
- Total runtime target: 5:45–6:15 (edit to exactly 6:00 if possible)
