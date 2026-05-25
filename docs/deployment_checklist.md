# Deployment Evidence Checklist

> This checklist covers what you need to demonstrate and the exact
> commands to produce each piece of evidence for grading.

---

## 1. Docker Compose — local run

### Pre-requisites
```bash
# 1. Train a model so mlruns/ exists
python -m src.training.train --sample

# 2. Build the FAISS index
python -m src.rag.build_index --sample

# 3. Confirm .env exists (never committed to Git)
#    Required keys: OPENAI_API_KEY or ANTHROPIC_API_KEY (for /ask-complaints)
#    Minimum viable .env:
#      MLFLOW_TRACKING_URI=file:/app/mlruns
#      MLFLOW_EXPERIMENT_NAME=customer-intelligence
```

### Build and run
```bash
# Build the image
docker compose build

# Start in detached mode
docker compose up -d

# Tail logs to confirm startup
docker compose logs -f ml-service
```

### Expected startup output
```
ml-service-1  | INFO: RAG retriever loaded -- /ask-complaints is available
ml-service-1  | INFO: Application startup complete.
ml-service-1  | INFO: Uvicorn running on http://0.0.0.0:8000
```

### Healthcheck
```bash
curl http://localhost:8000/health
# Expected:
# {"status":"ok","model_version":"XGBoost_stump_depth1",
#  "run_id":"...","is_ready":true,"rag_ready":true}
```

### Screenshot to capture
- Terminal showing `docker compose up` output with `Application startup complete`
- Browser showing `http://localhost:8000/docs` (FastAPI Swagger UI with all 5 endpoints)
- `curl http://localhost:8000/health` JSON response in terminal

### Teardown
```bash
docker compose down
```

---

## 2. Cloud / Remote Endpoint

### Option A — Railway free tier (recommended, no credit card)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link
railway login
railway init          # creates a Railway project
railway link

# Set environment variables via Railway dashboard or CLI
railway variables set MLFLOW_TRACKING_URI=file:/app/mlruns
railway variables set MLFLOW_EXPERIMENT_NAME=customer-intelligence

# Deploy (uses docker-compose.yml automatically)
railway up
```

Expected output: `Deployment live at https://customer-intelligence-<hash>.up.railway.app`

Test the live endpoint:
```bash
RAILWAY_URL="https://customer-intelligence-<hash>.up.railway.app"
curl "$RAILWAY_URL/health"
```

### Option B — Azure Container Instances (free tier, 180 vCPU-seconds/month)

```bash
# Build and push to Azure Container Registry
az acr build --registry <your-acr> --image customer-intelligence:v1 .

# Deploy as a container instance
az container create \
  --resource-group customer-intel-rg \
  --name ci-platform \
  --image <your-acr>.azurecr.io/customer-intelligence:v1 \
  --dns-name-label customer-intel-demo \
  --ports 8000 \
  --environment-variables \
      MLFLOW_TRACKING_URI=file:/app/mlruns \
      MLFLOW_EXPERIMENT_NAME=customer-intelligence

# Get the FQDN
az container show \
  --resource-group customer-intel-rg \
  --name ci-platform \
  --query ipAddress.fqdn
# → customer-intel-demo.<region>.azurecontainer.io
```

### Option C — localhost demo with port forwarding (minimal setup)
If cloud deployment is not feasible:

```bash
# Start the server directly
uvicorn src.serving.serve:app --host 0.0.0.0 --port 8000

# In another terminal, show the log file updating live
tail -f logs/customer_intel.jsonl | python -m json.tool
```

Evidence: capture a screen recording showing:
1. Server running in terminal 1
2. `curl` command in terminal 2 hitting `/customer-intel`
3. Log file updating in terminal 3

---

## 3. GitHub Actions Green Check

### Steps

1. Push all Phase 1–7 changes to `main` branch:
```bash
git add .
git commit -m "Phase 1-7: complete Customer Intelligence Platform"
git push origin main
```

2. Navigate to: `https://github.com/<your-username>/<repo-name>/actions`

3. Click the most recent workflow run — confirm all 4 jobs are green:
   - `lint` ✅
   - `unit-tests` ✅
   - `data-validation` ✅
   - `eval-gate` ✅

### Screenshot to capture
- The GitHub Actions workflow run page showing all 4 jobs with green checkmarks
- Click into `eval-gate` → expand the step "degraded model must be BLOCKED by
  promotion gate" → show the terminal output:
  ```
  [X] BLOCKED  --  PR-AUC delta -0.6027 < required +0.03 | F1 regressed -0.5425
  PASS: DummyClassifier_degraded correctly BLOCKED by promotion gate.
  ```

---

## 4. Sample curl commands for demo evidence

### /predict — valid input
```bash
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

### /predict — invalid input (triggers 422)
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 150, "job": "admin."}' | python -m json.tool
# Expected: HTTP 422 with validation error detail listing missing required fields
```

### /ask-complaints — evidence + refusal
```bash
# Evidence-based answer
curl -s -X POST http://localhost:8000/ask-complaints \
  -H "Content-Type: application/json" \
  -d '{"question": "What problems do customers report with mortgage payments?",
       "product": "Mortgage"}' | python -m json.tool

# Refusal (off-topic)
curl -s -X POST http://localhost:8000/ask-complaints \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current ECB interest rate policy?"}' \
  | python -m json.tool
# Expected: "refusal": true, "retrieved_ids": [], "answer": null
```

### /customer-intel — combined response
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
