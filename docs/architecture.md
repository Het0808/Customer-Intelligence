# System Architecture — Customer Intelligence Platform

> **How to use this document**
> Open Excalidraw (excalidraw.com) or draw.io (app.diagrams.net) and recreate the
> diagram described below. The components, lanes, and arrows map 1-to-1 to the
> codebase so a grader can trace every arrow to a source file.

---

## Canvas layout

Draw five horizontal **swim lanes**, top to bottom:

```
Lane 1 (blue)   : DATA INGESTION & VALIDATION
Lane 2 (green)  : ML PIPELINE
Lane 3 (orange) : RAG PIPELINE
Lane 4 (purple) : SERVING LAYER (FastAPI)
Lane 5 (red)    : CI/CD & MONITORING
```

---

## Lane 1 — Data Ingestion & Validation

### Components (left to right)

| Box | Label | Source file |
|-----|-------|-------------|
| Cylinder | `UCI Bank Marketing (41,163 rows)` | `data/raw/bank_marketing/` |
| Cylinder | `CFPB Complaints (≤10,000 rows)` | `data/raw/cfpb_complaints/` |
| Rectangle | `ingest.py` | `src/data_pipeline/ingest.py` |
| Rectangle | `validate.py` | `src/data_pipeline/validate.py` |
| Diamond | `Schema OK?` | pandera checks in `validate.py` |
| Cylinder | `data/samples/ (500 rows each)` | committed to Git |

### Arrows

```
UCI cylinder  ──►  ingest.py
CFPB cylinder ──►  ingest.py
ingest.py     ──►  validate.py
validate.py   ──►  Schema OK? (diamond)
Schema OK? YES ──► data/samples/ (cylinder, dashed = "committed to Git")
Schema OK? NO  ──► [red X label: "exit 1 / reject record"]
```

### Notes to add (small italic text)

- On `validate.py`: "pandera schema · 8 type checks · age 17-98 · y in {yes,no}"
- On `data/samples/`: "max 500 rows committed · no secrets in Git"

---

## Lane 2 — ML Pipeline

### Components (left to right)

| Box | Label | Source file |
|-----|-------|-------------|
| Rectangle | `features.py` | `src/data_pipeline/features.py` |
| Rectangle | `train.py` | `src/training/train.py` |
| Rectangle | `evaluate.py` (Promotion Gate) | `src/training/evaluate.py` |
| Cylinder | `MLflow Tracking Server` | `mlruns/` |
| Rectangle | `model_loader.py` | `src/serving/model_loader.py` |

### Arrows

```
data/samples/ ──►  features.py  [label: "60/20/20 stratified split"]
features.py   ──►  train.py     [label: "StandardScaler · OHE · 4 business features"]
train.py      ──►  evaluate.py  [label: "LogisticRegression baseline + XGBoost improved"]
evaluate.py   ──►  MLflow       [label: "PR-AUC, ROC-AUC, F1, confusion matrix, PR curve"]
evaluate.py   ──►  [diamond: "Gate PROMOTED?"]
diamond YES   ──►  MLflow       [label: "tag gate_decision=PROMOTED"]
diamond NO    ──►  [red X: "BLOCKED · exit 1"]
MLflow        ──►  model_loader.py  [label: "load best PROMOTED run"]
```

### Notes to add

- On `evaluate.py` gate: "Gate: ΔPR-AUC ≥ +3pp AND ΔF1 ≥ −2pp"
- On `train.py`: "XGBoost · n_estimators=300 · max_depth=5 · scale_pos_weight=8.09"
- Metrics box floating near MLflow: "val PR-AUC 0.7026 · val F1 0.6400 · val ROC-AUC 0.9245"

---

## Lane 3 — RAG Pipeline

### Components (left to right)

| Box | Label | Source file |
|-----|-------|-------------|
| Rectangle | `build_index.py` | `src/rag/build_index.py` |
| Rectangle | `Chunker` | inside `build_index.py` |
| Rectangle | `all-MiniLM-L6-v2` | `sentence_transformers` |
| Cylinder | `FAISS IndexFlatIP` | `faiss_index/index.bin + docstore.json` |
| Rectangle | `retrieve.py` | `src/rag/retrieve.py` |
| Diamond | `Score ≥ 0.35?` | `MIN_SCORE_THRESHOLD` in `retrieve.py` |
| Rectangle | `answer.py` | `src/rag/answer.py` |
| Rectangle | `LLM (Claude / local)` | called from `answer.py` |

### Arrows

```
CFPB cylinder ──►  build_index.py
build_index.py ──► Chunker       [label: "chunk_size=512, overlap=50"]
Chunker       ──►  all-MiniLM    [label: "384-dim embeddings · L2-normalised"]
all-MiniLM    ──►  FAISS index   [label: "IndexFlatIP · exact cosine search"]
query text    ──►  retrieve.py   [label: "embed query · pre-filter by product/issue/date"]
retrieve.py   ──►  FAISS index   [label: "top-k candidates"]
FAISS index   ──►  Score ≥ 0.35? (diamond)
diamond YES   ──►  answer.py     [label: "RetrievedChunks"]
diamond NO    ──►  [orange box: "refusal · insufficient_evidence=True"]
answer.py     ──►  LLM           [label: "evidence-grounded prompt"]
LLM           ──►  [output: "ComplaintAnswer"]
```

### Notes to add

- On FAISS: "500 chunks · 384-dim · ~0.7 MB on disk"
- On chunker: "sorted by complaint_id → deterministic rebuild"
- On retrieve.py: "Stage 1: metadata pre-filter → Stage 2: embed → Stage 3: cosine → Stage 4: refusal gate"
- On all-MiniLM: "CPU only · no API key · ~26ms/query"

---

## Lane 4 — Serving Layer

### Components (centre of lane)

```
┌─────────────────────────────────────────────────────────────┐
│               FastAPI  serve.py  (port 8000)                │
│                                                             │
│  GET /health          POST /predict         GET /metrics    │
│  POST /batch-score    POST /ask-complaints                  │
│  POST /customer-intel  ◄─── COMBINED ML + RAG              │
└─────────────────────────────────────────────────────────────┘
```

Draw this as a large rounded rectangle with six smaller rectangles inside it, one per endpoint.

**Highlight** `/customer-intel` with a thicker border or different fill to show it is the integration point.

### External arrows INTO the serving layer

```
model_loader.py  ──►  /predict, /batch-score, /customer-intel
retrieve.py      ──►  /ask-complaints, /customer-intel
```

### Internal arrows inside /customer-intel

```
/customer-intel ──► ML predict_proba()   [label: "conversion_probability"]
/customer-intel ──► retriever.retrieve() [label: "top-10 chunks · product filter"]
retrieve result ──► _build_complaint_themes()  [label: "group by CFPB issue taxonomy"]
both results    ──► CustomerIntelResponse
CustomerIntelResponse ──► _log_customer_intel()  [label: "JSONL audit · logs/customer_intel.jsonl"]
```

### Output arrows

```
/predict         ──► PredictionResponse (JSON)  [label: "probability · band · latency_ms"]
/ask-complaints  ──► ComplaintAnswer    (JSON)  [label: "answer · retrieved_ids · refusal"]
/customer-intel  ──► CustomerIntelResponse (JSON) [label: "conversion_band · complaint_themes · index_version"]
```

### Notes to add

- On Docker: "docker-compose up ml-service · healthcheck every 30s"
- On Pydantic: "schemas.py · 422 on invalid input · dot-notation aliases"
- On /customer-intel: "RAG is best-effort: index absent → complaint_themes=[]"

---

## Lane 5 — CI/CD & Monitoring

### CI/CD (left side of lane)

```
GitHub push/PR  ──►  [rectangle: "GitHub Actions  ci.yml"]
```

Inside the GitHub Actions rectangle draw **4 boxes** in a dependency graph:

```
lint  ──►  unit-tests  ──────────────────────►  eval-gate
      └──► data-validation  ─────────────────►  eval-gate
```

Label each box:
- `lint`: "ruff · E,F,W · --ignore E501,E402"
- `unit-tests`: "pytest 139 tests · <35s · no network"
- `data-validation`: "pandera sample + bad-record rejection"
- `eval-gate`: "DummyClassifier BLOCKED · exit 1 if PROMOTED"

### Monitoring (right side of lane)

```
[rectangle: ml_drift.py]
  ├── reference: first 250 rows of sample
  ├── current:  3 synthetic shifts (age+10, 15% nulls, blue-collar→services)
  ├── Evidently DataDriftPreset
  └── output: monitoring/reports/ml_drift_report.html
      + RETRAIN TRIGGERED if drift_intensity > 0.3

[rectangle: rag_monitor.py]
  ├── 10 EVAL_CASES from rag_eval.py
  ├── metrics: hit_rate, top1_score, refusal_rate, latency_ms
  └── output: monitoring/reports/rag_metrics.json
```

Draw arrows:

```
ml_drift.py ──►  [alarm icon: "RETRAIN TRIGGERED"]
RETRAIN     ──►  train.py  [dashed arrow: "re-run pipeline"]
rag_monitor.py ──► [dashboard icon: "rag_metrics.json"]
```

---

## Data flow summary (draw as a thin arrow spanning all 5 lanes)

```
Raw data (Lane 1)
  → Feature engineering (Lane 2)
  → Trained model + FAISS index (Lane 2/3)
  → Served via FastAPI (Lane 4)
  → Monitored for drift (Lane 5)
  → CI gate blocks regressions (Lane 5)
```

Draw this as a single thick curved arrow on the left margin of the diagram labelled
**"End-to-end data + model lifecycle"**.

---

## Colour legend (bottom right)

| Colour | Meaning |
|--------|---------|
| Blue   | Storage / data at rest |
| Green  | Compute / transformation |
| Orange | External API / model |
| Purple | Serving boundary |
| Red    | Failure / rejection path |
| Dashed | Optional / best-effort path |
